// Time-of-day model: sun/moon position, sky colors, fog and light levels.
// Colors are linear RGB; the renderer applies the same tone curve to sky and scene.
#pragma once
#include <cmath>
#include "vmath.h"

struct Environment {
    float hours = 0;
    V3 sunDir;          // toward the sun
    V3 moonDir;         // toward the moon (opposite the sun)
    V3 keyDir;          // dominant directional light: sun by day, moon by night
    V3 keyColor;        // its color times intensity
    V3 sunColor;        // sun disc / glow color
    V3 zenith, horizon; // sky gradient
    V3 ambientSky, ambientGround;
    float fogDensity = 0.008f;
    float day = 0;      // 0..1, full daylight
    float night = 1;    // 0..1, streetlights on, stars out
    float sunset = 0;   // 0..1, near-horizon glow
};

inline V3 mix3(V3 a, V3 b, float t) { return a + (b - a) * t; }

inline Environment environmentAt(float hours) {
    constexpr float PI = 3.14159265f;
    Environment e;
    e.hours = std::fmod(std::fmod(hours, 24.0f) + 24.0f, 24.0f);
    // Sun rises in the east at 06:00, peaks at 62 degrees at noon, sets at 18:00.
    float phase = (e.hours - 6.0f) / 12.0f * PI;
    float elev = std::sin(phase) * 62.0f * PI / 180.0f;
    float az = phase;
    const V3 east{1, 0, 0}, south{0, 0, -1}, up{0, 1, 0};
    e.sunDir = normalize(east * (std::cos(elev) * std::cos(az)) + south * (std::cos(elev) * std::sin(az) * 0.8f) + up * std::sin(elev));
    e.moonDir = normalize(V3{-e.sunDir.x, std::fabs(e.sunDir.y) * 0.8f + 0.25f, -e.sunDir.z});
    float deg = elev * 180.0f / PI;

    e.day = smoothstep(-4.0f, 12.0f, deg);
    e.night = 1.0f - smoothstep(-9.0f, 3.0f, deg);
    e.sunset = std::exp(-(deg / 7.0f) * (deg / 7.0f)) * smoothstep(-10.0f, -2.0f, deg);

    e.sunColor = mix3(V3{1.0f, 0.42f, 0.14f}, V3{1.0f, 0.95f, 0.88f}, smoothstep(0.0f, 25.0f, deg));
    float sunI = 2.1f * smoothstep(-2.0f, 8.0f, deg);
    V3 moon = V3{0.55f, 0.65f, 1.0f} * 0.05f;
    if (deg > -3.0f) {
        e.keyDir = e.sunDir.y > 0.02f ? e.sunDir : normalize(V3{e.sunDir.x, 0.02f, e.sunDir.z});
        e.keyColor = e.sunColor * sunI;
    } else {
        e.keyDir = e.moonDir;
        e.keyColor = moon * e.night;
    }

    e.zenith = mix3(V3{0.0015f, 0.0025f, 0.007f}, V3{0.10f, 0.25f, 0.65f}, e.day) + V3{0.05f, 0.03f, 0.08f} * e.sunset;
    e.horizon = mix3(V3{0.005f, 0.007f, 0.012f}, V3{0.50f, 0.62f, 0.80f}, e.day);
    e.horizon = mix3(e.horizon, V3{0.85f, 0.38f, 0.15f}, e.sunset * 0.8f);
    e.ambientSky = mix3(V3{0.020f, 0.026f, 0.045f}, e.zenith * 1.2f + V3{0.03f, 0.03f, 0.03f}, e.day);
    e.ambientGround = mix3(V3{0.006f, 0.007f, 0.010f}, e.horizon * 0.35f, e.day);
    e.fogDensity = lerpf(0.0085f, 0.0026f, e.day);
    return e;
}
