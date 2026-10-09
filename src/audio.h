// Car audio. Plays the mod's own FMOD Studio bank when the (proprietary, separately
// downloaded) FMOD Engine DLLs are present; otherwise a procedural fallback synth.
#pragma once
#include <atomic>
#include <memory>
#include <string>

struct SDL_AudioStream;

struct CarSound {
    float rpm = 900, maxRpm = 7000;
    float throttle = 0;   // 0..1
    float speedMs = 0;
    float slip = 0;       // 0..1
    float boost = 0;      // 0..1 turbo spool estimate
    int gear = 0;
    bool interior = true; // cockpit camera
};

class CarAudio {
public:
    CarAudio();
    ~CarAudio();
    // carDir: converted car folder (for sfx/*.bank). Returns false if no audio at all.
    bool init(const std::string& carDir, const std::string& carId);
    void shutdown();
    void update(const CarSound& s);
    bool usingFmod() const { return fmod != nullptr; }

private:
    static void callback(void* user, SDL_AudioStream* stream, int additional, int total);
    void synth(float* out, int frames);

    struct Fmod;
    std::unique_ptr<Fmod> fmod;
    SDL_AudioStream* stream = nullptr;
    std::atomic<float> rpm{900}, maxRpm{7000}, throttle{0}, speed{0}, slip{0};
    std::atomic<int> gear{0};
    // Synth state, only touched on the audio thread.
    double phase = 0, whinePhase = 0;
    float rpmSmooth = 900, thrSmooth = 0, slipSmooth = 0;
    float lp1 = 0, lp2 = 0, squealLp = 0, squealBp = 0, windLp = 0;
    unsigned noise = 12345;
};
