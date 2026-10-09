// Jolt Physics world: static track collision plus one player car driven by
// Jolt's WheeledVehicleController, configured from the converted AC car data.
// Jolt headers stay inside physics.cpp.
#pragma once
#include <memory>
#include <vector>
#include "assets.h"
#include "vmath.h"

struct CarInput {
    float throttle = 0;   // 0..1
    float brake = 0;      // 0..1
    float steer = 0;      // -1 (left) .. 1 (right)
    float handbrake = 0;  // 0..1
    int shift = 0;        // +1 up, -1 down (manual only), consumed each step
};

struct CarState {
    M4 body;              // body transform; origin at the center of gravity
    M4 wheels[4];         // LF, RF, LR, RR, oriented like the body at zero steer/spin
    V3 pos, vel, fwd, up, right;
    float speed = 0;      // forward speed, m/s
    float rpm = 0;
    float maxRpm = 7000;
    int gear = 0;         // -1 R, 0 N, 1..n
    bool manual = false;
    float slip = 0;       // 0..1 overall tire scrub, for audio and FX
    float suspensionVel = 0;
    float suspension[4] = {};  // current suspension length per wheel (m)
};

class Physics {
public:
    Physics();
    ~Physics();
    void addTrack(const std::vector<CollisionData>& meshes, const TrackInfo& info);
    void createCar(const CarSpec& spec, V3 pos, V3 fwd);
    void step(float dt, CarInput& input);
    void resetCar(V3 pos, V3 fwd);
    void setManual(bool manual);
    // Height of the first static surface below `from`, or NAN if nothing is hit.
    float groundBelow(V3 from, float maxDist = 50.0f) const;
    CarState state() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
