// initialopen: touge racing on converted Assetto Corsa content.
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_opengl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "assets.h"
#include "audio.h"
#include "physics.h"
#include "renderer.h"
#include "timing.h"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string track, car, spawn = "AC_PIT_0";
    std::string screenshot;  // render, save a BMP after `frames` frames, then quit
    int frames = 240;
    bool autodrive = false;  // follow the AI line (testing / future AI opponents)
    bool telemetry = false;
    float autodriveSpeed = 16.0f;  // m/s  // print body motion every 0.1 s (suspension tuning)
    bool chase = false;
    bool clean = false;  // start without the tape effect
    float timeOfDay = 23.0f;  // hours
};

// First folder under `root` (searched recursively) that contains `file`.
std::string findAsset(const fs::path& root, const char* file) {
    std::error_code ec;
    if (!fs::exists(root, ec)) return {};
    for (auto& e : fs::recursive_directory_iterator(root, ec))
        if (e.is_regular_file() && e.path().filename() == file) return e.path().parent_path().string();
    return {};
}

fs::path assetsRoot() {
    for (fs::path p : {fs::current_path() / "assets", fs::path(SDL_GetBasePath()) / "assets", fs::path(SDL_GetBasePath()) / ".." / "assets"})
        if (fs::exists(p)) return p;
    return fs::current_path() / "assets";
}

bool saveScreenshot(const std::string& path, int w, int h) {
    using ReadPixelsFn = void(APIENTRY*)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);
    auto readPixels = reinterpret_cast<ReadPixelsFn>(SDL_GL_GetProcAddress("glReadPixels"));
    if (!readPixels) return false;
    std::vector<uint8_t> px(size_t(w) * h * 4), flipped(px.size());
    readPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    for (int y = 0; y < h; ++y) std::copy_n(&px[size_t(h - 1 - y) * w * 4], size_t(w) * 4, &flipped[size_t(y) * w * 4]);
    SDL_Surface* s = SDL_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_ABGR8888, flipped.data(), w * 4);
    bool ok = s && SDL_SaveBMP(s, path.c_str());
    SDL_DestroySurface(s);
    return ok;
}

float approach(float v, float target, float rate, float dt) {
    float d = target - v, step = rate * dt;
    return std::fabs(d) <= step ? target : v + (d > 0 ? step : -step);
}

M4 rotationZ(float a) {
    M4 r = M4::identity();
    r.m[0] = std::cos(a); r.m[1] = std::sin(a);
    r.m[4] = -std::sin(a); r.m[5] = std::cos(a);
    return r;
}

M4 rotationOnly(const M4& m) {
    M4 r = m;
    r.m[12] = r.m[13] = r.m[14] = 0;
    return r;
}

int nearestAi(const std::vector<V3>& ai, V3 p) {
    int best = 0;
    float bd = 1e30f;
    for (int k = 0; k < int(ai.size()); ++k) {
        V3 d = ai[k] - p;
        float dd = d.x * d.x + d.z * d.z + d.y * d.y * 4.0f;
        if (dd < bd) { bd = dd; best = k; }
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--track") opt.track = next();
        else if (a == "--car") opt.car = next();
        else if (a == "--spawn") opt.spawn = next();
        else if (a == "--screenshot") opt.screenshot = next();
        else if (a == "--frames") opt.frames = std::max(1, std::atoi(next().c_str()));
        else if (a == "--autodrive") opt.autodrive = true;
        else if (a == "--autodrive-speed") opt.autodriveSpeed = float(std::atof(next().c_str()));
        else if (a == "--chase") opt.chase = true;
        else if (a == "--clean") opt.clean = true;
        else if (a == "--telemetry") opt.telemetry = true;
        else if (a == "--time") opt.timeOfDay = float(std::atof(next().c_str()));
        else {
            std::printf("usage: initialopen [--track DIR] [--car DIR] [--spawn AC_NODE] [--time HOURS] [--chase] [--clean] [--autodrive [--autodrive-speed M/S]] [--screenshot FILE.bmp [--frames N]]\n");
            return a == "--help" ? 0 : 1;
        }
    }

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD)) {
        std::fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    fs::path root = assetsRoot();
    if (opt.track.empty()) opt.track = findAsset(root / "tracks", "track.glb");
    if (opt.car.empty()) opt.car = findAsset(root / "cars", "car.glb");
    if (opt.track.empty() || opt.car.empty()) {
        std::fprintf(stderr, "No converted assets found under %s.\nConvert some first, see README (tools/acconv).\n", root.string().c_str());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_Window* window = SDL_CreateWindow("initialopen", 1280, 960, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    SDL_GLContext gl = window ? SDL_GL_CreateContext(window) : nullptr;
    if (!gl) {
        std::fprintf(stderr, "OpenGL 4.1 context: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_SetSwapInterval(opt.screenshot.empty() ? 1 : 0);

    Renderer renderer;
    if (!renderer.init()) {
        std::fprintf(stderr, "renderer init failed\n");
        return 1;
    }

    // --- Load content.
    auto t0 = std::chrono::steady_clock::now();
    std::string err;
    TrackInfo trackInfo;
    CarSpec carSpec;
    std::vector<CollisionData> collision;
    int trackModel = -1, carModel = -1;
    std::vector<MeshGroup> carGroups;
    std::vector<int> trackGroups;
    {
        ModelData track, car;
        if (!loadTrackInfo(opt.track + "/track.json", trackInfo, err) || !loadCarSpec(opt.car + "/car.json", carSpec, err) ||
            !loadGlb(opt.track + "/track.glb", track, &collision, err) || !loadGlb(opt.car + "/car.glb", car, nullptr, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        trackModel = renderer.addModel(track);
        carModel = renderer.addModel(car);
        for (int k = 0; k < int(track.groups.size()); ++k) trackGroups.push_back(k);
        carGroups = car.groups;
    }
    auto groupIndex = [&](const char* name) {
        for (int k = 0; k < int(carGroups.size()); ++k)
            if (carGroups[k].name == name) return k;
        return -1;
    };
    const int gBody = groupIndex("BODY"), gLights = groupIndex("LIGHTS_ON"), gBrake = groupIndex("LIGHTS_BRAKE");
    const int gSteer = groupIndex("STEER_HR");
    const char* wheelNames[4] = {"WHEEL_LF", "WHEEL_RF", "WHEEL_LR", "WHEEL_RR"};
    const char* suspNames[4] = {"SUSP_LF", "SUSP_RF", "SUSP_LR", "SUSP_RR"};
    int gWheel[4], gSusp[4];
    for (int k = 0; k < 4; ++k) {
        gWheel[k] = groupIndex(wheelNames[k]);
        gSusp[k] = groupIndex(suspNames[k]);
    }
    renderer.setLights(&trackInfo.lights);

    const float rideHeight = -carSpec.wheels[0].pos.y + carSpec.wheels[0].radius;  // CG above ground

    Physics physics;
    physics.addTrack(collision, trackInfo);
    collision.clear();
    // Place a point on the drivable surface below it (AC markers float a little above).
    auto onGround = [&](V3 p, bool& ok) {
        float g = physics.groundBelow(p + V3{0, 3, 0}, 10.0f);
        ok = !std::isnan(g);
        return ok ? V3{p.x, g + rideHeight + 0.05f, p.z} : p;
    };
    // Requested marker first, then the usual AC fallbacks; skip markers with no physics under them.
    V3 spawnPos{}, spawnFwd{0, 0, 1};
    std::string spawnName;
    std::vector<std::string> tried;
    for (const std::string& want : {opt.spawn, std::string("AC_PIT_0"), std::string("AC_HOTLAP_START_0"), std::string("AC_START_0")}) {
        if (std::find(tried.begin(), tried.end(), want) != tried.end()) continue;
        tried.push_back(want);
        const Spawn* s = trackInfo.findSpawn(want);
        bool ok = false;
        if (s) spawnPos = onGround(s->pos, ok);
        if (ok) {
            spawnFwd = s->fwd;
            spawnName = s->name;
            break;
        }
        if (s) std::printf("spawn %s has no collision under it, skipping\n", s->name.c_str());
    }
    if (spawnName.empty() && trackInfo.aiLine.size() > 1) {
        bool ok;
        spawnPos = onGround(trackInfo.aiLine[0], ok);
        spawnFwd = normalize(trackInfo.aiLine[1] - trackInfo.aiLine[0]);
        spawnName = "ai line start";
    }
    auto spawnAt = [&](V3 p) {
        bool ok;
        return onGround(p, ok);
    };
    std::printf("spawn: %s\n", spawnName.c_str());
    physics.createCar(carSpec, spawnPos, spawnFwd);
    double loadSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("Loaded '%s' + '%s' in %.2fs\n", trackInfo.name.c_str(), carSpec.name.c_str(), loadSec);

    // --- Time attack: progress along the racing line, lap timer, best-lap ghost.
    auto u8name = [](const std::string& s) {
        fs::path p(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
        auto u = p.filename().u8string();
        return std::string(u.begin(), u.end());
    };
    auto u8parent = [&](const std::string& s) {
        fs::path p(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
        auto u = p.parent_path().u8string();
        return u8name(std::string(u.begin(), u.end()));
    };
    const std::string trackKey = u8parent(opt.track) + "/" + u8name(opt.track), carId = u8name(opt.car);
    const std::string bestPath = "runs/" + trackKey + "/" + carId + ".best";
    TrackProgress progress;
    progress.init(trackInfo.aiLine);
    LapTimer timer;
    timer.init(progress.length(), progress.closed());
    Run best, current;
    bool haveBest = best.load(bestPath);
    if (haveBest) std::printf("best lap %s loaded from %s\n", formatLapTime(best.time).c_str(), bestPath.c_str());
    double lastSample = -1;
    float lastLap = 0, lapProgress = 0;
    bool lastLapValid = false, haveLastLap = false;
    std::string banner;
    double bannerUntil = 0;
    uint32_t bannerColor = 0;

    CarAudio audio;
    bool audioOk = opt.screenshot.empty() && audio.init(opt.car, fs::path(std::u8string(reinterpret_cast<const char8_t*>(opt.car.data()), opt.car.size())).filename().string());

    // --- State.
    CarInput input;
    bool cockpit = !opt.chase, lightsOn = true, manual = false, help = true, running = true;
    float timeOfDay = opt.timeOfDay;
    bool showSlider = true, dragging = false;
    // Time-of-day slider, in OSD (640x480) coordinates.
    const float sliderX0 = 200, sliderX1 = 440, sliderY = 32;
    auto toOsd = [&](float wx, float wy, float& ox, float& oy) {
        int w, h;
        SDL_GetWindowSize(window, &w, &h);
        float scale = std::min(w / 640.0f, h / 480.0f);
        ox = (wx - (w - 640.0f * scale) * 0.5f) / scale;
        oy = (wy - (h - 480.0f * scale) * 0.5f) / scale;
    };
    auto sliderHit = [&](float ox, float oy) { return showSlider && oy > sliderY - 12 && oy < sliderY + 16 && ox > sliderX0 - 8 && ox < sliderX1 + 8; };
    auto sliderSet = [&](float ox) { timeOfDay = std::clamp((ox - sliderX0) / (sliderX1 - sliderX0), 0.0f, 1.0f) * 24.0f; };
    float steerSmooth = 0, throttleSmooth = 0, brakeSmooth = 0;
    double simTime = 0, accumulator = 0;
    const double step = 1.0 / 120.0;
    float camShake = 0;
    // Driver's head: lags behind the shell under g-forces (sense of weight).
    V3 headOffset{}, prevVel{}, localAcc{};
    float lookYaw = 0;
    // Chase camera: car-relative offset on a spring, yaw that trails the car.
    V3 chaseOffset{}, chaseOffsetVel{};
    float chaseYaw = std::atan2(spawnFwd.x, spawnFwd.z), chaseFov = 52.0f, chaseRoll = 0;
    bool chaseInit = false;
    bool vhsOn = !opt.clean;
    bool ghostOn = true;
    // Scene resolution: native, half, or a retro 480 lines.
    int resMode = 0;
    const char* resNames[] = {"NATIVE", "HALF", "480P"};
    int frame = 0, aiHint = -1;
    SDL_Gamepad* pad = nullptr;
    auto last = std::chrono::steady_clock::now();

    while (running) {
        int shiftReq = 0;
        bool resetToRoad = false, resetToSpawn = false;
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
                case SDL_EVENT_QUIT: running = false; break;
                case SDL_EVENT_MOUSE_BUTTON_DOWN: {
                    float ox, oy;
                    toOsd(ev.button.x, ev.button.y, ox, oy);
                    if (ev.button.button == SDL_BUTTON_LEFT && sliderHit(ox, oy)) {
                        dragging = true;
                        sliderSet(ox);
                    }
                    break;
                }
                case SDL_EVENT_MOUSE_BUTTON_UP: dragging = false; break;
                case SDL_EVENT_MOUSE_MOTION:
                    if (dragging) {
                        float ox, oy;
                        toOsd(ev.motion.x, ev.motion.y, ox, oy);
                        sliderSet(ox);
                    }
                    break;
                case SDL_EVENT_GAMEPAD_ADDED:
                    if (!pad) pad = SDL_OpenGamepad(ev.gdevice.which);
                    break;
                case SDL_EVENT_GAMEPAD_REMOVED:
                    if (pad && SDL_GetGamepadID(pad) == ev.gdevice.which) { SDL_CloseGamepad(pad); pad = nullptr; }
                    break;
                case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
                    if (ev.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER) shiftReq = 1;
                    if (ev.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_SHOULDER) shiftReq = -1;
                    if (ev.gbutton.button == SDL_GAMEPAD_BUTTON_BACK) resetToRoad = true;
                    if (ev.gbutton.button == SDL_GAMEPAD_BUTTON_NORTH) cockpit = !cockpit;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (ev.key.repeat) break;
                    switch (ev.key.key) {
                        case SDLK_ESCAPE: running = false; break;
                        case SDLK_C: cockpit = !cockpit; break;
                        case SDLK_H: lightsOn = !lightsOn; break;
                        case SDLK_T: manual = !manual; physics.setManual(manual); break;
                        case SDLK_E: shiftReq = 1; break;
                        case SDLK_Q: shiftReq = -1; break;
                        case SDLK_R: resetToRoad = true; break;
                        case SDLK_P: resetToSpawn = true; break;
                        case SDLK_F1: help = !help; break;
                        case SDLK_F2: showSlider = !showSlider; break;
                        case SDLK_V: vhsOn = !vhsOn; break;
                        case SDLK_F3: resMode = (resMode + 1) % 3; break;
                        case SDLK_G: ghostOn = !ghostOn; break;
                        default: break;
                    }
                    break;
                default: break;
            }
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::min(0.1f, std::chrono::duration<float>(now - last).count());
        last = now;
        if (!opt.screenshot.empty()) dt = 1.0f / 60.0f;  // deterministic test runs

        CarState car = physics.state();

        // --- Input: keyboard (ramped) and gamepad (analog).
        const bool* keys = SDL_GetKeyboardState(nullptr);
        // Hold [ or ] to scrub time: three hours per second.
        timeOfDay += (float(keys[SDL_SCANCODE_RIGHTBRACKET]) - float(keys[SDL_SCANCODE_LEFTBRACKET])) * dt * 3.0f;
        timeOfDay = std::fmod(timeOfDay + 24.0f, 24.0f);
        float steerT = float(keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT]) - float(keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT]);
        float thrT = float(keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP]);
        float brkT = float(keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN]);
        float hbT = float(keys[SDL_SCANCODE_SPACE]);
        bool highBeam = keys[SDL_SCANCODE_L];
        steerSmooth = approach(steerSmooth, steerT, steerT == 0 ? 4.5f : 2.2f, dt);
        throttleSmooth = approach(throttleSmooth, thrT, 4.0f, dt);
        brakeSmooth = approach(brakeSmooth, brkT, 5.0f, dt);
        input.steer = steerSmooth;
        input.throttle = throttleSmooth;
        input.brake = brakeSmooth;
        input.handbrake = hbT;
        if (pad) {
            float sx = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTX) / 32767.0f;
            // Response curve: fine control around center, full lock still at the stop.
            if (std::fabs(sx) > 0.08f) input.steer = std::copysign(std::pow(std::min((std::fabs(sx) - 0.08f) / 0.92f, 1.0f), 1.6f), sx);
            float rt = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) / 32767.0f;
            float lt = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) / 32767.0f;
            input.throttle = std::max(input.throttle, rt);
            input.brake = std::max(input.brake, lt);
            if (SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_SOUTH)) input.handbrake = 1;
            if (SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_EAST)) highBeam = true;
        }
        if (opt.autodrive && !trackInfo.aiLine.empty()) {
            // Pure pursuit along the racing line at a cautious pace.
            aiHint = nearestAi(trackInfo.aiLine, car.pos);
            V3 target = trackInfo.aiLine[(aiHint + 6) % trackInfo.aiLine.size()];
            V3 to = normalize(V3{target.x - car.pos.x, 0, target.z - car.pos.z});
            float side = dot(to, car.right), ahead = dot(to, car.fwd);
            input.steer = std::clamp(std::atan2(side, ahead) * 2.5f, -1.0f, 1.0f);
            float wanted = opt.autodriveSpeed;
            input.throttle = car.speed < wanted ? 0.7f : 0.0f;
            input.brake = car.speed > wanted + 4.0f ? 0.4f : 0.0f;
        }
        input.shift += shiftReq;

        if (resetToRoad && !trackInfo.aiLine.empty()) {
            int k = nearestAi(trackInfo.aiLine, car.pos);
            V3 a = trackInfo.aiLine[k], b = trackInfo.aiLine[(k + 1) % trackInfo.aiLine.size()];
            physics.resetCar(spawnAt(a), normalize(V3{b.x - a.x, 0, b.z - a.z}));
            timer.invalidate();
        }
        if (resetToSpawn) {
            physics.resetCar(spawnPos, spawnFwd);
            timer.stop();
        }

        // --- Fixed-step physics.
        accumulator += dt;
        int steps = 0;
        while (accumulator >= step && steps < 12) {
            physics.step(float(step), input);
            accumulator -= step;
            simTime += step;
            ++steps;
        }
        car = physics.state();

        // --- Lap timing and recording.
        lapProgress = progress.project(car.pos);
        LapTimer::Event lapEvent = timer.update(lapProgress, simTime);
        if (lapEvent == LapTimer::Event::Finished) {
            current.time = timer.finishedTime();
            std::copy(timer.finishedSplits(), timer.finishedSplits() + LapTimer::kSectors, current.sectors);
            lastLap = current.time;
            lastLapValid = timer.finishedValid();
            haveLastLap = true;
            bool improved = lastLapValid && (!haveBest || current.time < best.time);
            std::printf("lap %s%s%s\n", formatLapTime(current.time).c_str(), lastLapValid ? "" : " (invalid)", improved ? " NEW BEST" : "");
            if (improved) {
                current.track = trackKey;
                current.car = carId;
                best = current;
                haveBest = true;
                if (!best.save(bestPath)) std::printf("could not save %s\n", bestPath.c_str());
            }
            banner = (improved ? "NEW BEST " : (lastLapValid ? "LAP " : "INVALID LAP ")) + formatLapTime(current.time);
            bannerColor = improved ? rgba(90, 255, 140) : (lastLapValid ? rgba(235, 235, 235) : rgba(255, 80, 60));
            bannerUntil = simTime + 4.0;
        }
        if (lapEvent == LapTimer::Event::Started || (lapEvent == LapTimer::Event::Finished && timer.running())) {
            current.frames.clear();
            lastSample = -1;
        }
        if (timer.running() && (lastSample < 0 || simTime - lastSample >= 1.0 / 30.0)) {
            lastSample = simTime;
            RunFrame f;
            f.t = timer.lapTime(simTime);
            f.progress = lapProgress;
            f.pos = car.pos;
            f.rot = quatFromMat(car.body);
            M4 inv = rigidInverse(car.body);
            for (int w = 0; w < 4; ++w) {
                M4 rel = inv * car.wheels[w];
                f.wheelPos[w] = {rel.m[12], rel.m[13], rel.m[14]};
                f.wheelRot[w] = quatFromMat(rel);
            }
            f.speed = car.speed;
            f.rpm = car.rpm;
            f.steer = input.steer;
            f.throttle = input.throttle;
            f.brake = input.brake;
            f.gear = car.gear;
            current.frames.push_back(f);
        }

        if (audioOk) {
            CarSound snd;
            snd.rpm = car.rpm;
            snd.maxRpm = car.maxRpm;
            snd.throttle = input.throttle;
            snd.speedMs = car.speed;
            snd.slip = car.slip;
            snd.boost = std::clamp(input.throttle * (car.rpm / car.maxRpm) * 1.4f - 0.2f, 0.0f, 1.0f);
            snd.gear = car.gear;
            snd.interior = cockpit;
            audio.update(snd);
        }
        if (opt.telemetry && int(simTime * 10) != int((simTime - steps * step) * 10)) {
            float pitch = std::asin(std::clamp(car.fwd.y, -1.0f, 1.0f)) * 57.3f;
            float roll = std::asin(std::clamp(car.right.y, -1.0f, 1.0f)) * 57.3f;
            std::printf("T %5.1f y %8.3f vy %6.2f pitch %6.2f roll %6.2f kmh %5.1f susp %.3f %.3f %.3f %.3f\n", simTime, car.pos.y,
                        car.vel.y, pitch, roll, car.speed * 3.6f, car.suspension[0], car.suspension[1], car.suspension[2],
                        car.suspension[3]);
        }

        // Body-space acceleration (+X = left), updated whenever physics stepped.
        if (steps > 0) {
            V3 accel = (car.vel - prevVel) * (1.0f / float(steps * step));
            V3 a{-dot(accel, car.right), dot(accel, car.up), dot(accel, car.fwd)};
            localAcc = localAcc + (a - localAcc) * 0.25f;
        }

        // --- Camera.
        int fbW = 0, fbH = 0;
        SDL_GetWindowSizeInPixels(window, &fbW, &fbH);
        const M4& B = car.body;
        RenderView rv;
        rv.time = float(simTime);
        camShake = camShake * 0.92f + (car.suspensionVel * 0.0012f + std::fabs(car.speed) * 0.00003f + car.slip * 0.0015f) * 0.08f;
        float t = float(simTime);
        V3 jitter{std::sin(t * 61.0f) * camShake, std::sin(t * 47.0f + 1.3f) * camShake, 0};
        if (cockpit) {
            // Roll-cage mount: rigid with the shell, so pitch, roll and vibration come through.
            // Body-space acceleration pushes the head the other way (braking leans you
            // forward, corners push you outward), then a soft spring brings it back.
            if (steps > 0) {
                float pdt = float(steps * step);
                V3 headTarget{clampf(-localAcc.x * 0.006f, -0.07f, 0.07f), clampf(-localAcc.y * 0.002f, -0.03f, 0.03f),
                              clampf(-localAcc.z * 0.005f, -0.06f, 0.06f)};
                headOffset = headOffset + (headTarget - headOffset) * std::min(1.0f, pdt * 6.0f);
            }
            // Look slightly into the direction of travel when the car rotates or slides.
            V3 flatVel{car.vel.x, 0, car.vel.z};
            float slideYaw = length(flatVel) > 3.0f ? std::atan2(dot(flatVel, car.right) * -1.0f, std::max(dot(flatVel, car.fwd), 0.5f)) : 0.0f;
            lookYaw += (clampf(slideYaw * 0.35f, -0.25f, 0.25f) - lookYaw) * std::min(1.0f, dt * 4.0f);
            V3 eye = B.transformPoint(carSpec.driverEyes + headOffset + jitter);
            float pitch = carSpec.onboardPitchDeg * 3.14159f / 180.0f;
            V3 look = normalize(B.transformDir({std::sin(lookYaw), std::sin(pitch), std::cos(lookYaw) * std::cos(pitch)}));
            rv.view = lookAt(eye, eye + look, B.transformDir({0, 1, 0}));
            rv.proj = perspective(56.0f * 3.14159f / 180.0f, float(fbW) / float(std::max(fbH, 1)), 0.05f, 2000.0f);
            rv.camPos = eye;
        } else {
            // Heading the camera wants: the car's nose, swung toward the direction of
            // travel as speed builds, so drifts show the car sideways.
            V3 flatVel{car.vel.x, 0, car.vel.z};
            float spd = length(flatVel);
            float carYaw = std::atan2(car.fwd.x, car.fwd.z);
            float velYaw = spd > 1.0f ? std::atan2(flatVel.x, flatVel.z) : carYaw;
            float slide = std::remainder(velYaw - carYaw, 6.2831853f);
            if (dot(flatVel, car.fwd) < 0) slide = 0;  // reversing: stay behind the car
            float targetYaw = carYaw + slide * 0.55f * clampf(spd / 12.0f, 0.0f, 1.0f);
            float camDt = std::min(dt, 0.05f);
            if (!chaseInit) chaseYaw = targetYaw;
            chaseYaw += std::remainder(targetYaw - chaseYaw, 6.2831853f) * std::min(1.0f, camDt * 3.5f);
            V3 dir{std::sin(chaseYaw), 0, std::cos(chaseYaw)};

            // Offset springs: speed stretches the distance, throttle pulls the car away,
            // braking lets it come closer, cornering swings the camera outward.
            V3 sideL{dir.z, 0, -dir.x};  // left of the camera heading
            float dist = 5.6f + spd * 0.035f - clampf(localAcc.z, -12.0f, 12.0f) * 0.07f;
            float height = 1.9f - clampf(spd * 0.006f, 0.0f, 0.35f);
            V3 want = dir * -dist + V3{0, height, 0} + sideL * clampf(-localAcc.x * 0.05f, -0.8f, 0.8f);
            if (!chaseInit) {
                chaseOffset = want;
                chaseInit = true;
            }
            const float k = 7.0f;  // spring rate (1/s), critically damped
            chaseOffsetVel = chaseOffsetVel + ((want - chaseOffset) * (k * k) - chaseOffsetVel * (2.0f * k)) * camDt;
            chaseOffset = chaseOffset + chaseOffsetVel * camDt;
            V3 eye = car.pos + chaseOffset + jitter * 3.0f;
            // Never dig into the hillside.
            float ground = physics.groundBelow(eye + V3{0, 3.0f, 0}, 8.0f);
            if (!std::isnan(ground)) eye.y = std::max(eye.y, ground + 0.6f);

            V3 target = car.pos + V3{0, 0.75f, 0} + flatVel * 0.12f;
            chaseRoll += (clampf(localAcc.x * 0.006f, -0.06f, 0.06f) - chaseRoll) * std::min(1.0f, camDt * 3.0f);
            V3 fwdView = normalize(target - eye);
            V3 rightView = normalize(cross(fwdView, V3{0, 1, 0}));
            V3 upView = normalize(V3{0, 1, 0} * std::cos(chaseRoll) + rightView * std::sin(chaseRoll));
            chaseFov += (clampf(52.0f + spd * 0.22f, 52.0f, 66.0f) - chaseFov) * std::min(1.0f, camDt * 2.0f);
            rv.view = lookAt(eye, target, upView);
            rv.proj = perspective(chaseFov * 3.14159f / 180.0f, float(fbW) / float(std::max(fbH, 1)), 0.1f, 2000.0f);
            rv.camPos = eye;
        }
        prevVel = car.vel;
        rv.lightsOn = lightsOn;
        rv.vhs = vhsOn ? 1.0f : 0.0f;
        rv.renderScale = resMode == 0 ? 1.0f : (resMode == 1 ? 0.5f : 480.0f / float(std::max(fbH, 1)));
        rv.env = environmentAt(timeOfDay);
        rv.highBeam = highBeam;
        float hx = carSpec.hullMax.x * 0.65f, hy = carSpec.hullMin.y + 0.45f, hz = carSpec.hullMax.z - 0.25f;
        rv.headlightPos[0] = B.transformPoint({hx, hy, hz});
        rv.headlightPos[1] = B.transformPoint({-hx, hy, hz});
        rv.beamDir = normalize(car.fwd - car.up * (highBeam ? 0.0f : 0.07f));

        // --- Scene.
        for (int g : trackGroups) renderer.submit(trackModel, g, M4::identity());
        renderer.submit(carModel, gBody, B);
        renderer.submit(carModel, gLights, B, lightsOn ? 1.0f : 0.0f);
        renderer.submit(carModel, gBrake, B, input.brake > 0.05f ? 1.0f : (lightsOn ? 0.3f : 0.0f));
        if (gSteer >= 0) renderer.submit(carModel, gSteer, B * carGroups[gSteer].matrix * rotationZ(input.steer * carSpec.steerLockDeg * 3.14159f / 180.0f));
        for (int k = 0; k < 4; ++k) {
            if (gWheel[k] >= 0) renderer.submit(carModel, gWheel[k], car.wheels[k]);
            if (gSusp[k] >= 0 && gWheel[k] >= 0) {
                const M4& ws = carGroups[gWheel[k]].matrix;
                const M4& ss = carGroups[gSusp[k]].matrix;
                V3 wp{car.wheels[k].m[12], car.wheels[k].m[13], car.wheels[k].m[14]};
                V3 off{ss.m[12] - ws.m[12], ss.m[13] - ws.m[13], ss.m[14] - ws.m[14]};
                renderer.submit(carModel, gSusp[k], translation(wp) * rotationOnly(B) * translation(off));
            }
        }

        // Ghost of the best lap, time-synchronized with the current one.
        if (ghostOn && haveBest && timer.running() && timer.lapTime(simTime) <= best.time) {
            RunFrame gf = best.sample(timer.lapTime(simTime));
            M4 G = matFromQuat(gf.rot, gf.pos);
            float alpha = 0.38f * smoothstep(2.5f, 9.0f, length(gf.pos - rv.camPos));
            if (alpha > 0.01f) {
                renderer.submit(carModel, gBody, G, 1.0f, alpha);
                renderer.submit(carModel, gLights, G, lightsOn ? 1.0f : 0.0f, alpha);
                for (int k = 0; k < 4; ++k)
                    if (gWheel[k] >= 0) renderer.submit(carModel, gWheel[k], G * matFromQuat(gf.wheelRot[k], gf.wheelPos[k]), 1.0f, alpha);
            }
        }

        // --- Camcorder OSD and gauges.
        Osd& osd = renderer.osd();
        osd.clear();
        char buf[96];
        uint32_t white = rgba(235, 235, 235);
        float kmh = std::fabs(car.speed) * 3.6f;
        osd.gauge(70, 400, 46, car.rpm / 10000.0f, car.maxRpm / 10000.0f, 10, "RPM X1000");
        osd.gauge(170, 400, 46, kmh / 300.0f, 2.0f, 6, "KM/H");
        std::snprintf(buf, sizeof buf, "%c", car.gear < 0 ? 'R' : (car.gear == 0 ? 'N' : char('0' + car.gear)));
        osd.rect(222, 360, 26, 30, rgba(4, 6, 5, 230));
        osd.text(228, 364, buf, rgba(90, 255, 140), 2.0f, false);
        osd.text(222, 394, manual ? "MT" : "AT", rgba(90, 255, 140), 1.0f, false);
        std::snprintf(buf, sizeof buf, "%3d", int(kmh));
        osd.text(146, 418, buf, rgba(90, 255, 140), 1.0f, false);
        if (showSlider) {
            uint32_t dim = rgba(235, 235, 235, 150);
            osd.rect(sliderX0, sliderY, sliderX1 - sliderX0, 2, dim);
            for (int hr = 0; hr <= 24; hr += 6) osd.rect(sliderX0 + (sliderX1 - sliderX0) * hr / 24.0f - 0.5f, sliderY - 3, 1, 8, dim);
            float kx = sliderX0 + (sliderX1 - sliderX0) * timeOfDay / 24.0f;
            osd.rect(kx - 3, sliderY - 6, 6, 14, white);
            std::snprintf(buf, sizeof buf, "TIME %02d:%02d", int(timeOfDay) % 24, int(timeOfDay * 60) % 60);
            osd.text(286, sliderY + 12, buf, white, 1.0f);
        }
        // Time attack HUD (top right) and live delta to the best lap (top center).
        {
            const uint32_t green = rgba(90, 255, 140), red = rgba(255, 80, 60), dim = rgba(200, 200, 200);
            float lt = timer.lapTime(simTime);
            std::string cur = timer.running() ? formatLapTime(lt) : "-:--.---";
            osd.text(452, 58, cur, timer.running() && !timer.isValid() ? red : white, 2.0f);
            osd.text(452, 78, "BEST " + (haveBest ? formatLapTime(best.time) : std::string("-:--.---")), dim, 1.0f);
            osd.text(452, 90, "LAST " + (haveLastLap ? formatLapTime(lastLap) : std::string("-:--.---")) + (haveLastLap && !lastLapValid ? " X" : ""),
                     dim, 1.0f);
            if (timer.running()) {
                float y = 102;
                for (int k = 0; k < timer.sector() && k < LapTimer::kSectors; ++k) {
                    float t = timer.split(k), prevT = k > 0 ? timer.split(k - 1) : 0.0f;
                    std::snprintf(buf, sizeof buf, "S%d %s", k + 1, formatLapTime(t - prevT).c_str());
                    uint32_t c = dim;
                    if (haveBest) {
                        float d = t - best.sectors[k];
                        std::snprintf(buf + std::strlen(buf), sizeof buf - std::strlen(buf), " %+.2f", d);
                        c = d <= 0 ? green : red;
                    }
                    osd.text(452, y, buf, c, 1.0f);
                    y += 12;
                }
                if (!timer.isValid()) osd.text(452, y, "INVALID", red, 1.0f);
                if (haveBest && lt <= best.time + 30.0f) {
                    float d = lt - best.timeAtProgress(lapProgress);
                    std::snprintf(buf, sizeof buf, "%+.2f", d);
                    osd.text(320 - float(std::strlen(buf)) * 6.0f, 60, buf, d <= 0 ? green : red, 2.0f);
                }
            } else {
                const char* hint = "TIME ATTACK: CROSS THE START LINE";
                osd.text(320 - float(std::strlen(hint)) * 3.0f, 60, hint, dim, 1.0f);
            }
            if (simTime < bannerUntil) osd.text(320 - float(banner.size()) * 6.0f, 150, banner, bannerColor, 2.0f);
        }
        if (help && simTime < 25.0) {
            osd.text(28, 60, trackInfo.name, white, 1.0f);
            osd.text(28, 72, carSpec.name, white, 1.0f);
            osd.text(28, 92, "WASD DRIVE  SPACE HANDBRAKE  L HIGH BEAM  H LIGHTS", white, 1.0f);
            osd.text(28, 104, "C CAMERA  T AT/MT  Q/E SHIFT  R RESET TO ROAD  P PIT  F1 HELP", white, 1.0f);
            osd.text(28, 116, "DRAG SLIDER OR HOLD [ ] TO CHANGE TIME  F2 HIDE SLIDER  V TAPE EFFECT", white, 1.0f);
            std::snprintf(buf, sizeof buf, "F3 RESOLUTION: %s   G GHOST %s", resNames[resMode], ghostOn ? "ON" : "OFF");
            osd.text(28, 128, buf, white, 1.0f);
        }

        renderer.render(rv, fbW, fbH);
        if (!opt.screenshot.empty() && ++frame >= opt.frames) {
            bool ok = saveScreenshot(opt.screenshot, fbW, fbH);
            std::printf("t=%.1fs pos=(%.1f %.1f %.1f) speed=%.1f km/h rpm=%.0f gear=%d ai=%d screenshot=%s\n", simTime, car.pos.x,
                        car.pos.y, car.pos.z, kmh, car.rpm, car.gear, aiHint, ok ? opt.screenshot.c_str() : "FAILED");
            running = false;
        }
        SDL_GL_SwapWindow(window);
    }

    audio.shutdown();
    renderer.shutdown();
    if (pad) SDL_CloseGamepad(pad);
    SDL_GL_DestroyContext(gl);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
