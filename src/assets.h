// Loading of converted assets (see tools/acconv): .glb models and .json metadata.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "vmath.h"

struct ImageData {
    enum Format { RGBA8, BC1, BC2, BC3 } format = RGBA8;
    int width = 0, height = 0;
    std::vector<std::vector<uint8_t>> mips;  // level 0 first
};

struct MaterialData {
    enum Alpha { Opaque, Mask, Blend } alpha = Opaque;
    int image = -1;
    int detailImage = -1;  // AC multimap: diffuse alpha blends toward this texture
    float detailUv = 1.0f;
    float baseColor[4] = {1, 1, 1, 1};
    float emissive[3] = {0, 0, 0};
    float cutoff = 0.5f;
    bool doubleSided = false;
    std::string name;
};

struct Primitive {
    uint32_t firstIndex = 0, indexCount = 0;
    int material = -1;
    V3 bmin, bmax;
};

// One glb node with a mesh. `matrix` is the node transform (column-major).
struct MeshGroup {
    std::string name;
    M4 matrix = M4::identity();
    std::vector<Primitive> prims;
};

struct ModelData {
    std::vector<float> pos, nrm, uv;  // 3, 3, 2 floats per vertex
    std::vector<uint32_t> idx;
    std::vector<MaterialData> materials;
    std::vector<ImageData> images;
    std::vector<MeshGroup> groups;
    int findGroup(const std::string& name) const;
};

struct CollisionData {
    std::string surface;
    std::vector<V3> v;
    std::vector<uint32_t> i;
};

// Collision meshes (nodes named COLLISION_<SURFACE>) go to `collision` instead of `out`.
bool loadGlb(const std::string& path, ModelData& out, std::vector<CollisionData>* collision, std::string& err);

struct Spawn {
    std::string name;
    V3 pos, fwd;
};

struct TrackLight {
    V3 pos;
    V3 color;
    float range;
};

struct TrackInfo {
    std::string name;
    float length = 0;
    std::map<std::string, float> friction;  // per surface key
    std::vector<Spawn> spawns;
    std::vector<TrackLight> lights;
    std::vector<V3> aiLine;
    const Spawn* findSpawn(const std::string& prefix) const;
};

struct WheelSpec {
    std::string name;
    V3 pos;  // wheel center at design ride height, body space
    float radius = 0.3f, width = 0.2f;
    bool steer = false;
    float springRate = 60000, dampBump = 3000, dampRebound = 4000;
    float packerRange = 0.1f;
    float dx0 = 1.2f, dy0 = 1.2f, frictionLimitAngle = 8.0f;
};

struct CarSpec {
    std::string name;
    float mass = 1200;
    V3 inertiaBox{1.8f, 1.2f, 4.2f};
    V3 driverEyes{0.4f, 0.6f, 0};
    float onboardPitchDeg = 0;
    V3 hullMin{-0.9f, -0.4f, -2.2f}, hullMax{0.9f, 0.8f, 2.2f};
    WheelSpec wheels[4];  // LF, RF, LR, RR
    float arbFront = 0, arbRear = 0;
    float steerLockDeg = 400, steerRatio = 14;
    float idleRpm = 900, limiterRpm = 7000, engineInertia = 0.1f;
    std::vector<std::pair<float, float>> torqueCurve;  // rpm -> Nm, turbo boost applied
    std::string traction = "RWD";
    std::vector<float> gears;
    float reverseGear = -3.0f, finalDrive = 4.0f;
    float diffPower = 0.3f;
    float shiftUpMs = 200;
    float brakeTorque = 2000, brakeFrontShare = 0.6f, handbrakeTorque = 800;
};

bool loadTrackInfo(const std::string& path, TrackInfo& out, std::string& err);
bool loadCarSpec(const std::string& path, CarSpec& out, std::string& err);
