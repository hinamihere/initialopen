#include "audio.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {
constexpr int kRate = 44100;
constexpr double kTau = 6.283185307179586;

std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower((unsigned char)c));
    return s;
}
}  // namespace

// ---------------------------------------------------------------------------
// FMOD Studio, loaded at runtime. FMOD is proprietary, so it is never linked or
// shipped: players put fmod.dll + fmodstudio.dll next to the executable (or in
// ./fmod/). Only the small C API subset below is used.

struct CarAudio::Fmod {
    using Result = int;
    struct Guid {
        uint32_t d1;
        uint16_t d2, d3;
        uint8_t d4[8];
    };
    struct ParamDesc {  // FMOD_STUDIO_PARAMETER_DESCRIPTION, padded for newer versions
        const char* name;
        uint32_t id[2];
        float minimum, maximum, defaultValue;
        int type;
        uint32_t flags;
        uint8_t reserved[64];
    };

    SDL_SharedObject *core = nullptr, *studio = nullptr;
    Result (*SystemCreate)(void**, unsigned) = nullptr;
    Result (*SystemInitialize)(void*, int, unsigned, unsigned, void*) = nullptr;
    Result (*SystemLoadBankFile)(void*, const char*, unsigned, void**) = nullptr;
    Result (*SystemGetEventByID)(void*, const Guid*, void**) = nullptr;
    Result (*SystemUpdate)(void*) = nullptr;
    Result (*SystemRelease)(void*) = nullptr;
    Result (*ParseID)(const char*, Guid*) = nullptr;
    Result (*DescCreateInstance)(void*, void**) = nullptr;
    Result (*DescParamCount)(void*, int*) = nullptr;
    Result (*DescParamByIndex)(void*, int, ParamDesc*) = nullptr;
    Result (*InstStart)(void*) = nullptr;
    Result (*InstSetParam)(void*, const char*, float, int) = nullptr;
    Result (*InstSetVolume)(void*, float) = nullptr;

    void* system = nullptr;

    // One playing event and how its parameters map onto car state.
    struct Event {
        void* inst = nullptr;
        enum Kind { Rpm, Throttle, Boost, Speed, Slip, Gear, Other };
        struct Param {
            std::string name;
            Kind kind;
            float lo, hi;
        };
        std::vector<Param> params;
    };
    std::map<std::string, Event> events;  // by short name, e.g. "engine_ext"
    int lastGear = 0;

    template <class F> bool sym(SDL_SharedObject* lib, F& fn, const char* name) {
        fn = reinterpret_cast<F>(SDL_LoadFunction(lib, name));
        return fn != nullptr;
    }

    bool load() {
        namespace fs = std::filesystem;
        fs::path base = fs::path(std::u8string(reinterpret_cast<const char8_t*>(SDL_GetBasePath())));
        for (fs::path dir : {base, base / "fmod"}) {
            auto u8 = [](const fs::path& p) { auto s = p.u8string(); return std::string(s.begin(), s.end()); };
            if (!fs::exists(dir / "fmodstudio.dll")) continue;
            core = SDL_LoadObject(u8(dir / "fmod.dll").c_str());  // dependency of fmodstudio.dll
            studio = SDL_LoadObject(u8(dir / "fmodstudio.dll").c_str());
            if (studio) break;
        }
        if (!studio) return false;
        bool ok = sym(studio, SystemCreate, "FMOD_Studio_System_Create") && sym(studio, SystemInitialize, "FMOD_Studio_System_Initialize") &&
                  sym(studio, SystemLoadBankFile, "FMOD_Studio_System_LoadBankFile") &&
                  sym(studio, SystemGetEventByID, "FMOD_Studio_System_GetEventByID") && sym(studio, SystemUpdate, "FMOD_Studio_System_Update") &&
                  sym(studio, SystemRelease, "FMOD_Studio_System_Release") && sym(studio, ParseID, "FMOD_Studio_ParseID") &&
                  sym(studio, DescCreateInstance, "FMOD_Studio_EventDescription_CreateInstance") &&
                  sym(studio, DescParamCount, "FMOD_Studio_EventDescription_GetParameterDescriptionCount") &&
                  sym(studio, DescParamByIndex, "FMOD_Studio_EventDescription_GetParameterDescriptionByIndex") &&
                  sym(studio, InstStart, "FMOD_Studio_EventInstance_Start") &&
                  sym(studio, InstSetParam, "FMOD_Studio_EventInstance_SetParameterByName") &&
                  sym(studio, InstSetVolume, "FMOD_Studio_EventInstance_SetVolume");
        if (!ok) {
            std::printf("audio: fmodstudio.dll found but is not an FMOD 2.x Studio API build\n");
            return false;
        }
        // The header version must match the DLL; probe 2.00.00 .. 2.04.40 (BCD-style).
        for (int minor = 4; minor >= 0 && !system; --minor)
            for (int patch = 40; patch >= 0 && !system; --patch) {
                unsigned v = 0x00020000u | unsigned(minor) << 8 | unsigned((patch / 10) << 4 | (patch % 10));
                void* s = nullptr;
                if (SystemCreate(&s, v) == 0) system = s;
            }
        if (!system) {
            std::printf("audio: could not create an FMOD Studio system (unsupported version?)\n");
            return false;
        }
        if (SystemInitialize(system, 128, 0, 0, nullptr) != 0) {
            std::printf("audio: FMOD Studio initialize failed\n");
            return false;
        }
        return true;
    }

    bool loadCar(const std::string& carDir, const std::string& carId) {
        namespace fs = std::filesystem;
        fs::path sfx = fs::path(std::u8string(reinterpret_cast<const char8_t*>(carDir.data()), carDir.size())) / "sfx";
        std::string bankPath;
        std::error_code ec;
        for (auto& e : fs::directory_iterator(sfx, ec))
            if (lower(e.path().extension().string()) == ".bank") {
                auto s = e.path().u8string();
                bankPath.assign(s.begin(), s.end());
            }
        if (bankPath.empty()) return false;
        void* bank = nullptr;
        Result r = SystemLoadBankFile(system, bankPath.c_str(), 0, &bank);
        if (r != 0) {
            std::printf("audio: FMOD could not load %s (error %d)\n", bankPath.c_str(), r);
            return false;
        }
        // GUIDs.txt maps "{guid} event:/cars/<car>/<name>"; there is no strings bank.
        std::ifstream g(sfx / "GUIDs.txt");
        std::string line;
        const std::string prefix = "event:/cars/" + lower(carId) + "/";
        while (std::getline(g, line)) {
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string guidStr = line.substr(0, sp), path = line.substr(sp + 1);
            while (!path.empty() && (path.back() == '\r' || path.back() == ' ')) path.pop_back();
            if (lower(path).rfind(prefix, 0) != 0) continue;
            std::string shortName = path.substr(prefix.size());
            static const char* wanted[] = {"engine_ext", "engine_int", "turbo", "skid_ext", "skid_int", "transmission", "wind", "limiter"};
            if (std::find_if(std::begin(wanted), std::end(wanted), [&](const char* w) { return shortName == w; }) == std::end(wanted))
                continue;
            Guid id;
            void* desc = nullptr;
            if (ParseID(guidStr.c_str(), &id) != 0 || SystemGetEventByID(system, &id, &desc) != 0) continue;
            Event ev;
            if (DescCreateInstance(desc, &ev.inst) != 0) continue;
            int count = 0;
            DescParamCount(desc, &count);
            std::string list;
            for (int k = 0; k < count; ++k) {
                ParamDesc pd{};
                if (DescParamByIndex(desc, k, &pd) != 0 || !pd.name) continue;
                std::string n = lower(pd.name);
                Event::Kind kind = n.find("rpm") != std::string::npos                                    ? Event::Rpm
                                   : (n.find("throttle") != std::string::npos || n.find("gas") != std::string::npos ||
                                      n.find("load") != std::string::npos)                                 ? Event::Throttle
                                   : n.find("boost") != std::string::npos || n.find("turbo") != std::string::npos ? Event::Boost
                                   : n.find("speed") != std::string::npos || n.find("kmh") != std::string::npos   ? Event::Speed
                                   : (n.find("slip") != std::string::npos || n.find("skid") != std::string::npos) ? Event::Slip
                                   : n.find("gear") != std::string::npos                                           ? Event::Gear
                                                                                                                   : Event::Other;
                ev.params.push_back({pd.name, kind, pd.minimum, pd.maximum});
                list += std::string(list.empty() ? "" : ", ") + pd.name + "[" + std::to_string(int(pd.minimum)) + ".." +
                        std::to_string(int(pd.maximum)) + "]";
            }
            std::printf("audio: %s params: %s\n", shortName.c_str(), list.c_str());
            InstStart(ev.inst);
            events[shortName] = std::move(ev);
        }
        if (events.empty()) {
            std::printf("audio: no playable car events found in the bank\n");
            return false;
        }
        return true;
    }

    void update(const CarSound& s) {
        for (auto& [name, ev] : events) {
            for (const auto& p : ev.params) {
                float v;
                switch (p.kind) {
                    case Event::Rpm: v = s.rpm; break;
                    case Event::Throttle: v = p.lo + (p.hi - p.lo) * s.throttle; break;
                    case Event::Boost: v = p.lo + (p.hi - p.lo) * s.boost; break;
                    case Event::Speed: v = p.hi > 100 ? std::fabs(s.speedMs) * 3.6f : std::fabs(s.speedMs); break;
                    case Event::Slip: v = p.lo + (p.hi - p.lo) * s.slip; break;
                    case Event::Gear: v = float(s.gear); break;
                    default: continue;
                }
                InstSetParam(ev.inst, p.name.c_str(), std::clamp(v, p.lo, p.hi), 0);
            }
            // Interior vs exterior variants follow the camera.
            float vol = 1.0f;
            if (name.ends_with("_ext")) vol = s.interior ? 0.0f : 1.0f;
            if (name.ends_with("_int")) vol = s.interior ? 1.0f : 0.0f;
            if (name == "limiter") vol = s.rpm > s.maxRpm * 0.985f ? 1.0f : 0.0f;
            InstSetVolume(ev.inst, vol);
        }
        SystemUpdate(system);
    }

    ~Fmod() {
        if (system) SystemRelease(system);
        if (studio) SDL_UnloadObject(studio);
        if (core) SDL_UnloadObject(core);
    }
};

CarAudio::CarAudio() = default;
CarAudio::~CarAudio() { shutdown(); }

bool CarAudio::init(const std::string& carDir, const std::string& carId) {
    auto f = std::make_unique<Fmod>();
    if (f->load() && f->loadCar(carDir, carId)) {
        fmod = std::move(f);
        std::printf("audio: playing the car's FMOD sound bank\n");
        return true;
    }
    std::printf("audio: FMOD not available, using the fallback synth (see README to enable real car sounds)\n");
    SDL_AudioSpec spec = {SDL_AUDIO_F32, 1, kRate};
    stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, &CarAudio::callback, this);
    if (!stream) return false;
    SDL_ResumeAudioStreamDevice(stream);
    return true;
}

void CarAudio::shutdown() {
    fmod.reset();
    if (stream) SDL_DestroyAudioStream(stream);
    stream = nullptr;
}

void CarAudio::update(const CarSound& s) {
    if (fmod) {
        fmod->update(s);
        return;
    }
    rpm = s.rpm;
    maxRpm = s.maxRpm;
    throttle = s.throttle;
    speed = s.speedMs;
    slip = s.slip;
    gear = s.gear;
}

void CarAudio::callback(void* user, SDL_AudioStream* stream, int additional, int) {
    auto* self = static_cast<CarAudio*>(user);
    int frames = additional / int(sizeof(float));
    if (frames <= 0) return;
    std::vector<float> buf(static_cast<size_t>(frames));
    self->synth(buf.data(), frames);
    SDL_PutAudioStreamData(stream, buf.data(), frames * int(sizeof(float)));
}

void CarAudio::synth(float* out, int frames) {
    const float targetRpm = rpm.load(), mr = std::max(maxRpm.load(), 1000.0f), thr = throttle.load();
    const float spd = std::fabs(speed.load()), sl = slip.load();
    const int g = gear.load();
    auto rnd = [this]() {
        noise = noise * 1664525u + 1013904223u;
        return float(noise >> 8) / float(1 << 24) * 2.0f - 1.0f;
    };
    for (int i = 0; i < frames; ++i) {
        rpmSmooth += (targetRpm - rpmSmooth) * 0.0015f;
        thrSmooth += (thr - thrSmooth) * 0.002f;
        slipSmooth += (sl - slipSmooth) * 0.001f;

        // Inline-6 four-stroke: three firing pulses per crank revolution.
        double fire = rpmSmooth / 60.0 * 3.0;
        phase += fire / kRate;
        if (phase > 1.0) phase -= 1.0;
        float p = float(phase);
        float pulse = std::exp(-p * 9.0f) - 0.11f;               // sharp combustion pulse
        float body = std::sin(float(kTau) * p) * 0.5f + std::sin(float(kTau) * 0.5f * p) * 0.35f;
        float raw = pulse * (0.55f + 0.6f * thrSmooth) + body * 0.35f + rnd() * 0.08f * thrSmooth;
        // Two one-pole low-passes: the exhaust gets brighter as revs rise.
        float cutoff = 0.05f + 0.25f * (rpmSmooth / mr);
        lp1 += (raw - lp1) * cutoff;
        lp2 += (lp1 - lp2) * cutoff;
        float engine = lp2 * (0.35f + 0.45f * thrSmooth);

        // Straight-cut gear whine rises with road speed and is louder in low gears.
        whinePhase += (spd * 38.0 + 1.0) / kRate;
        if (whinePhase > 1.0) whinePhase -= 1.0;
        float whine = std::sin(float(kTau * whinePhase)) * std::min(spd / 30.0f, 1.0f) * (g > 0 ? 0.05f / float(g) : 0.02f);

        // Tyre squeal: band-limited noise around ~900 Hz.
        float n = rnd();
        squealLp += (n - squealLp) * 0.15f;
        squealBp += (squealLp - squealBp) * 0.08f;
        float squeal = (squealLp - squealBp) * slipSmooth * slipSmooth * 1.4f;

        windLp += (rnd() - windLp) * 0.02f;
        float wind = windLp * std::min(spd / 50.0f, 1.0f) * 0.5f;

        out[i] = std::clamp((engine + whine + squeal + wind) * 0.8f, -1.0f, 1.0f);
    }
}
