// Procedural car audio on an SDL3 audio stream: engine, straight-cut gear whine,
// tyre squeal and wind. Stand-in until a sample-based engine sound system exists.
#pragma once
#include <atomic>

struct SDL_AudioStream;

class CarAudio {
public:
    bool init();
    void shutdown();
    void update(float rpm, float maxRpm, float throttle, float speedMs, float slip, int gear);

private:
    static void callback(void* user, SDL_AudioStream* stream, int additional, int total);
    void synth(float* out, int frames);

    SDL_AudioStream* stream = nullptr;
    std::atomic<float> rpm{900}, maxRpm{7000}, throttle{0}, speed{0}, slip{0};
    std::atomic<int> gear{0};
    // Synth state, only touched on the audio thread.
    double phase = 0, whinePhase = 0;
    float rpmSmooth = 900, thrSmooth = 0, slipSmooth = 0;
    float lp1 = 0, lp2 = 0, squealLp = 0, squealBp = 0, windLp = 0;
    unsigned noise = 12345;
};
