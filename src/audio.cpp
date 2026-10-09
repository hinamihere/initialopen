#include "audio.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
constexpr int kRate = 44100;
constexpr double kTau = 6.283185307179586;
}  // namespace

bool CarAudio::init() {
    SDL_AudioSpec spec = {SDL_AUDIO_F32, 1, kRate};
    stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, &CarAudio::callback, this);
    if (!stream) return false;
    SDL_ResumeAudioStreamDevice(stream);
    return true;
}

void CarAudio::shutdown() {
    if (stream) SDL_DestroyAudioStream(stream);
    stream = nullptr;
}

void CarAudio::update(float r, float mr, float t, float s, float sl, int g) {
    rpm = r;
    maxRpm = mr;
    throttle = t;
    speed = s;
    slip = sl;
    gear = g;
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
