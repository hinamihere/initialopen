#include "physics.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Vehicle/VehicleCollisionTester.h>
#include <Jolt/Physics/Vehicle/VehicleConstraint.h>
#include <Jolt/Physics/Vehicle/WheeledVehicleController.h>
#include <Jolt/RegisterTypes.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>

using namespace JPH;

namespace {

namespace Layers {
constexpr ObjectLayer STATIC = 0;
constexpr ObjectLayer MOVING = 1;
}  // namespace Layers

class BPLayers final : public BroadPhaseLayerInterface {
public:
    uint GetNumBroadPhaseLayers() const override { return 2; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer layer) const override { return BroadPhaseLayer(BroadPhaseLayer::Type(layer)); }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(BroadPhaseLayer layer) const override {
        return layer.GetValue() == 0 ? "STATIC" : "MOVING";
    }
#endif
};

class ObjVsBP final : public ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(ObjectLayer a, BroadPhaseLayer b) const override {
        return a == Layers::MOVING || b.GetValue() == Layers::MOVING;
    }
};

class ObjPair final : public ObjectLayerPairFilter {
public:
    bool ShouldCollide(ObjectLayer a, ObjectLayer b) const override { return a == Layers::MOVING || b == Layers::MOVING; }
};

void toM4(const Mat44& m, M4& out) { m.StoreFloat4x4(reinterpret_cast<Float4*>(out.m)); }
V3 toV3(Vec3 v) { return {v.GetX(), v.GetY(), v.GetZ()}; }
Vec3 toJ(V3 v) { return Vec3(v.x, v.y, v.z); }

// Suspension "feel" normalization. Many AC mods ship race setups (2+ Hz springs,
// a few cm of travel, huge anti-roll bars) that feel harsh and hide weight transfer.
// Every car gets the same ride character from its own mass and weight split, so
// duels stay fair and the body visibly pitches and rolls.
struct SuspensionFeel {
    float rideFreqFront = 1.45f;  // Hz, body bounce frequency per axle
    float rideFreqRear = 1.55f;   // slightly higher at the rear to damp pitch ("flat ride")
    float dampingRatio = 0.40f;   // fraction of critical damping
    float bumpTravel = 0.10f;     // m of compression available from ride height
    float arbScale = 0.30f;       // multiplier on the mod's anti-roll bar rates
};
constexpr SuspensionFeel kFeel;

// Map AC's differential lock (0 = open, 1 = locked) onto Jolt's max/min wheel speed ratio.
float limitedSlipRatio(float power) {
    float open = 1.0f - std::clamp(power, 0.0f, 1.0f);
    return 1.05f + 4.0f * open * open;
}

}  // namespace

struct Physics::Impl {
    std::unique_ptr<TempAllocatorImpl> temp;
    std::unique_ptr<JobSystemThreadPool> jobs;
    BPLayers bpLayers;
    ObjVsBP objVsBp;
    ObjPair objPair;
    PhysicsSystem system;
    Body* car = nullptr;
    Ref<VehicleConstraint> vehicle;
    WheeledVehicleController* controller = nullptr;
    float maxSteer = 0.5f;
    float limiter = 7000;
    bool manual = false;
    float prevSusp[4] = {};
    float suspVel = 0;

    Impl() {
        RegisterDefaultAllocator();
        Factory::sInstance = new Factory();
        RegisterTypes();
        temp = std::make_unique<TempAllocatorImpl>(32 * 1024 * 1024);
        int threads = std::max(1, int(std::thread::hardware_concurrency()) - 1);
        jobs = std::make_unique<JobSystemThreadPool>(cMaxPhysicsJobs, cMaxPhysicsBarriers, threads);
        system.Init(1024, 0, 4096, 4096, bpLayers, objVsBp, objPair);
    }

    ~Impl() {
        if (vehicle) {
            system.RemoveStepListener(vehicle);
            system.RemoveConstraint(vehicle);
        }
        vehicle = nullptr;
        UnregisterTypes();
        delete Factory::sInstance;
        Factory::sInstance = nullptr;
    }
};

Physics::Physics() : impl(std::make_unique<Impl>()) {}
Physics::~Physics() = default;

void Physics::addTrack(const std::vector<CollisionData>& meshes, const TrackInfo& info) {
    BodyInterface& bi = impl->system.GetBodyInterface();
    for (const CollisionData& cm : meshes) {
        if (cm.i.size() < 3) continue;
        VertexList verts;
        verts.reserve(cm.v.size());
        for (const V3& p : cm.v) verts.push_back(Float3(p.x, p.y, p.z));
        IndexedTriangleList tris;
        tris.reserve(cm.i.size() / 3);
        for (size_t k = 0; k + 2 < cm.i.size(); k += 3) tris.push_back(IndexedTriangle(cm.i[k], cm.i[k + 1], cm.i[k + 2], 0));
        MeshShapeSettings settings(std::move(verts), std::move(tris));
        ShapeSettings::ShapeResult res = settings.Create();
        if (res.HasError()) {
            std::fprintf(stderr, "collision mesh %s: %s\n", cm.surface.c_str(), res.GetError().c_str());
            continue;
        }
        BodyCreationSettings bcs(res.Get(), RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::STATIC);
        auto it = info.friction.find(cm.surface);
        bcs.mFriction = it != info.friction.end() ? it->second : 0.8f;
        bi.CreateAndAddBody(bcs, EActivation::DontActivate);
        std::printf("collision %-6s %7zu tris, friction %.2f\n", cm.surface.c_str(), cm.i.size() / 3, bcs.mFriction);
    }
    impl->system.OptimizeBroadPhase();
}

void Physics::createCar(const CarSpec& spec, V3 pos, V3 fwd) {
    Impl& I = *impl;
    I.limiter = spec.limiterRpm;

    // Body: the AC collider hull as a box, with the center of mass at the body origin
    // (converted car data is already in center-of-gravity space).
    V3 c = (spec.hullMin + spec.hullMax) * 0.5f, h = (spec.hullMax - spec.hullMin) * 0.5f;
    RefConst<Shape> hull = RotatedTranslatedShapeSettings(toJ(c), Quat::sIdentity(), new BoxShapeSettings(toJ(h) * 0.95f)).Create().Get();
    RefConst<Shape> chassis = OffsetCenterOfMassShapeSettings(-hull->GetCenterOfMass(), hull).Create().Get();
    BodyCreationSettings bcs(chassis, RVec3(pos.x, pos.y, pos.z), Quat::sRotation(Vec3::sAxisY(), std::atan2(fwd.x, fwd.z)),
                             EMotionType::Dynamic, Layers::MOVING);
    MassProperties mp;
    mp.SetMassAndInertiaOfSolidBox(toJ(spec.inertiaBox), 1000.0f);
    mp.ScaleToMass(spec.mass);
    bcs.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
    bcs.mMassPropertiesOverride = mp;
    bcs.mFriction = 0.3f;
    bcs.mLinearDamping = 0.0f;
    bcs.mAngularDamping = 0.05f;
    I.car = I.system.GetBodyInterface().CreateBody(bcs);
    I.system.GetBodyInterface().AddBody(I.car->GetID(), EActivation::Activate);

    VehicleConstraintSettings vs;
    vs.mMaxPitchRollAngle = DegreesToRadians(75.0f);
    // Arcade allowance: at least 30 degrees of lock so big slides stay catchable.
    I.maxSteer = DegreesToRadians(std::max(spec.steerLockDeg / std::max(spec.steerRatio, 1.0f), 30.0f));
    float zf = spec.wheels[0].pos.z, zr = spec.wheels[2].pos.z;
    float frontShare = std::clamp(-zr / std::max(zf - zr, 0.1f), 0.2f, 0.8f);
    for (int w = 0; w < 4; ++w) {
        const WheelSpec& s = spec.wheels[w];
        bool front = w < 2;
        float cornerMass = spec.mass * (front ? frontShare : 1.0f - frontShare) * 0.5f;
        float omega = 2.0f * JPH_PI * (front ? kFeel.rideFreqFront : kFeel.rideFreqRear);
        float stiffness = cornerMass * omega * omega;
        float damping = 2.0f * kFeel.dampingRatio * std::sqrt(stiffness * cornerMass);
        float sag = cornerMass * 9.81f / stiffness;
        float rest = kFeel.bumpTravel + 0.02f;  // suspension length at ride height
        auto* ws = new WheelSettingsWV;  // owned by vs.mWheels
        ws->mPosition = Vec3(s.pos.x, s.pos.y + rest, s.pos.z);
        ws->mSuspensionMinLength = rest - kFeel.bumpTravel;
        ws->mSuspensionMaxLength = rest + sag;  // spring reaches its natural length at full droop
        ws->mSuspensionSpring.mMode = ESpringMode::StiffnessAndDamping;
        ws->mSuspensionSpring.mStiffness = stiffness;
        ws->mSuspensionSpring.mDamping = damping;
        ws->mRadius = s.radius;
        ws->mWidth = s.width;
        ws->mMaxSteerAngle = s.steer ? I.maxSteer : 0.0f;
        ws->mMaxBrakeTorque = spec.brakeTorque * (front ? spec.brakeFrontShare : 1.0f - spec.brakeFrontShare);
        ws->mMaxHandBrakeTorque = front ? 0.0f : std::max(spec.handbrakeTorque, 1500.0f);
        ws->mInertia = 1.2f;
        // Grip peaks at the tyre's friction limit angle, then falls off gently so
        // slides stay controllable instead of snapping.
        float a = std::max(s.frictionLimitAngle, 3.0f);
        ws->mLateralFriction.Clear();
        ws->mLateralFriction.AddPoint(0.0f, 0.0f);
        ws->mLateralFriction.AddPoint(a, s.dy0);
        ws->mLateralFriction.AddPoint(a * 2.5f, s.dy0 * 0.93f);
        ws->mLateralFriction.AddPoint(40.0f, s.dy0 * 0.82f);
        ws->mLateralFriction.AddPoint(90.0f, s.dy0 * 0.75f);
        ws->mLongitudinalFriction.Clear();
        ws->mLongitudinalFriction.AddPoint(0.0f, 0.0f);
        ws->mLongitudinalFriction.AddPoint(0.1f, s.dx0);
        ws->mLongitudinalFriction.AddPoint(0.3f, s.dx0 * 0.9f);
        ws->mLongitudinalFriction.AddPoint(1.0f, s.dx0 * 0.78f);
        vs.mWheels.push_back(ws);
    }
    vs.mAntiRollBars.resize(2);
    vs.mAntiRollBars[0].mLeftWheel = 0;
    vs.mAntiRollBars[0].mRightWheel = 1;
    vs.mAntiRollBars[0].mStiffness = spec.arbFront * kFeel.arbScale;
    vs.mAntiRollBars[1].mLeftWheel = 2;
    vs.mAntiRollBars[1].mRightWheel = 3;
    vs.mAntiRollBars[1].mStiffness = spec.arbRear * kFeel.arbScale;

    auto* cs = new WheeledVehicleControllerSettings;
    float peak = 1.0f;
    for (auto& p : spec.torqueCurve) peak = std::max(peak, p.second);
    cs->mEngine.mMaxTorque = peak;
    cs->mEngine.mMinRPM = spec.idleRpm;
    cs->mEngine.mMaxRPM = spec.limiterRpm;
    cs->mEngine.mInertia = std::max(spec.engineInertia, 0.05f);
    cs->mEngine.mNormalizedTorque.Clear();
    for (auto& p : spec.torqueCurve) {
        float x = (p.first - spec.idleRpm) / std::max(spec.limiterRpm - spec.idleRpm, 1.0f);
        if (x >= 0.0f && x <= 1.0f) cs->mEngine.mNormalizedTorque.AddPoint(x, p.second / peak);
    }
    cs->mTransmission.mMode = ETransmissionMode::Auto;
    cs->mTransmission.mGearRatios.clear();
    for (float g : spec.gears) cs->mTransmission.mGearRatios.push_back(g);
    cs->mTransmission.mReverseGearRatios = {spec.reverseGear};
    cs->mTransmission.mShiftUpRPM = spec.limiterRpm * 0.93f;
    cs->mTransmission.mShiftDownRPM = spec.limiterRpm * 0.5f;
    cs->mTransmission.mSwitchTime = spec.shiftUpMs / 1000.0f;
    cs->mTransmission.mClutchReleaseTime = 0.15f;
    cs->mTransmission.mClutchStrength = 20.0f;

    float lsr = limitedSlipRatio(spec.diffPower);
    auto addDiff = [&](int left, int right, float share) {
        VehicleDifferentialSettings d;
        d.mLeftWheel = left;
        d.mRightWheel = right;
        d.mDifferentialRatio = spec.finalDrive;
        d.mLimitedSlipRatio = lsr;
        d.mEngineTorqueRatio = share;
        cs->mDifferentials.push_back(d);
    };
    if (spec.traction == "FWD") {
        addDiff(0, 1, 1.0f);
    } else if (spec.traction == "AWD") {
        // Rear-biased split; the center LSD shifts torque forward when the rears spin,
        // roughly how ATTESA-style systems behave.
        addDiff(0, 1, 0.25f);
        addDiff(2, 3, 0.75f);
        cs->mDifferentialLimitedSlipRatio = 1.4f;
    } else {
        addDiff(2, 3, 1.0f);
    }
    vs.mController = cs;

    I.vehicle = new VehicleConstraint(*I.car, vs);
    I.vehicle->SetVehicleCollisionTester(new VehicleCollisionTesterCastCylinder(Layers::MOVING, 0.05f));
    // Assetto Corsa style: tyre grip times surface grip.
    I.vehicle->SetCombineFriction([](uint, float& lon, float& lat, const Body& other, const SubShapeID&) {
        lon *= other.GetFriction();
        lat *= other.GetFriction();
    });
    I.system.AddConstraint(I.vehicle);
    I.system.AddStepListener(I.vehicle);
    I.controller = static_cast<WheeledVehicleController*>(I.vehicle->GetController());
}

void Physics::setManual(bool manual) {
    impl->manual = manual;
    VehicleTransmission& tr = impl->controller->GetTransmission();
    tr.mMode = manual ? ETransmissionMode::Manual : ETransmissionMode::Auto;
    if (manual && tr.GetCurrentGear() == 0) tr.Set(1, 1.0f);
}

float Physics::groundBelow(V3 from, float maxDist) const {
    RRayCast ray(RVec3(from.x, from.y, from.z), Vec3(0, -maxDist, 0));
    RayCastResult hit;
    if (!impl->system.GetNarrowPhaseQuery().CastRay(ray, hit)) return NAN;
    return from.y - maxDist * hit.mFraction;
}

void Physics::resetCar(V3 pos, V3 fwd) {
    BodyInterface& bi = impl->system.GetBodyInterface();
    bi.SetPositionAndRotation(impl->car->GetID(), RVec3(pos.x, pos.y, pos.z), Quat::sRotation(Vec3::sAxisY(), std::atan2(fwd.x, fwd.z)),
                              EActivation::Activate);
    bi.SetLinearAndAngularVelocity(impl->car->GetID(), Vec3::sZero(), Vec3::sZero());
    impl->controller->GetEngine().SetCurrentRPM(1500.0f);
    if (impl->manual) impl->controller->GetTransmission().Set(1, 1.0f);
}

void Physics::step(float dt, CarInput& in) {
    Impl& I = *impl;
    BodyInterface& bi = I.system.GetBodyInterface();
    const BodyID id = I.car->GetID();
    Mat44 xf = I.car->GetWorldTransform();
    Vec3 fwd = xf.GetAxisZ(), right = -xf.GetAxisX();
    Vec3 vel = I.car->GetLinearVelocity();
    float vFwd = vel.Dot(fwd), vRight = vel.Dot(right);
    float speed = vel.Length();

    // Steering: less lock at speed, plus a little automatic counter-steer toward
    // the direction of travel so slides feel controllable on a pad or keyboard.
    float lock = I.maxSteer * (1.0f - 0.6f * std::clamp(speed / 45.0f, 0.0f, 1.0f));
    float desired = in.steer * lock;
    if (speed > 4.0f && vFwd > 0) desired += std::atan2(vRight, std::max(vFwd, 0.1f)) * 0.45f;
    float rightIn = std::clamp(desired / I.maxSteer, -1.0f, 1.0f);

    float forward = in.throttle, brake = in.brake;
    VehicleTransmission& tr = I.controller->GetTransmission();
    if (I.manual) {
        int top = int(tr.mGearRatios.size());
        if (in.shift != 0) tr.Set(std::clamp(tr.GetCurrentGear() + in.shift, -1, top), 1.0f);
        bool parked = std::fabs(vFwd) < 1.5f && in.throttle < 0.05f;  // no creeping at a standstill
        tr.Set(tr.GetCurrentGear(), parked ? 0.0f : 1.0f);
    } else if (in.brake > 0.1f && in.throttle < 0.05f && vFwd < 1.0f) {
        forward = -in.brake;  // holding brake when stopped engages reverse
        brake = 0;
    }
    in.shift = 0;

    if (forward != 0 || brake != 0 || in.handbrake != 0 || rightIn != 0) bi.ActivateBody(id);
    I.controller->SetDriverInput(forward, rightIn, brake, in.handbrake);
    bi.AddForce(id, -vel * (0.45f * speed));  // aero drag

    I.system.Update(dt, 1, I.temp.get(), I.jobs.get());

    float acc = 0;
    for (uint w = 0; w < 4; ++w) {
        float len = I.vehicle->GetWheel(w)->GetSuspensionLength();
        acc += std::fabs(len - I.prevSusp[w]) / dt;
        I.prevSusp[w] = len;
    }
    I.suspVel = acc * 0.25f;
}

CarState Physics::state() const {
    const Impl& I = *impl;
    CarState s;
    Mat44 xf = I.car->GetWorldTransform();
    toM4(xf, s.body);
    // Wheel meshes are modelled in body orientation, where vehicle right is -X.
    for (uint w = 0; w < 4; ++w) toM4(I.vehicle->GetWheelWorldTransform(w, -Vec3::sAxisX(), Vec3::sAxisY()), s.wheels[w]);
    s.pos = toV3(xf.GetTranslation());
    s.vel = toV3(I.car->GetLinearVelocity());
    s.fwd = toV3(xf.GetAxisZ());
    s.up = toV3(xf.GetAxisY());
    s.right = toV3(-xf.GetAxisX());
    s.speed = dot(s.vel, s.fwd);
    s.rpm = I.controller->GetEngine().GetCurrentRPM();
    s.maxRpm = I.limiter;
    s.gear = I.controller->GetTransmission().GetCurrentGear();
    s.manual = I.manual;
    float slip = 0;
    for (uint w = 0; w < 4; ++w) {
        const auto* wheel = static_cast<const WheelWV*>(I.vehicle->GetWheel(w));
        if (!wheel->HasContact()) continue;
        float lat = std::fabs(wheel->mLateralSlip) / DegreesToRadians(25.0f);
        float lon = std::fabs(wheel->mLongitudinalSlip) / 0.6f;
        slip = std::max(slip, std::max(lat, lon));
    }
    float speedFade = std::clamp(std::sqrt(s.vel.x * s.vel.x + s.vel.z * s.vel.z) / 4.0f, 0.0f, 1.0f);
    s.slip = std::clamp(slip, 0.0f, 1.0f) * speedFade;
    s.suspensionVel = I.suspVel;
    for (uint w = 0; w < 4; ++w) s.suspension[w] = I.vehicle->GetWheel(w)->GetSuspensionLength();
    return s;
}
