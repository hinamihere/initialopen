#include "renderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "sokol_gfx.h"
#include "sokol_log.h"

namespace {

// ---------------------------------------------------------------------------
// Shaders (GLSL 4.1 core). Uniforms are vec4 arrays, matching sokol's std140 path.

constexpr int kMaxLights = 16;
constexpr int kFrameVec4 = 10 + 2 * kMaxLights + 2;
constexpr int kMatVec4 = 5;
constexpr int kShadowVec4 = 9;
constexpr int kShadowSize = 2048;
constexpr int kSkyVec4 = 8;

const char* kSceneVS = R"(#version 410
uniform vec4 vs_params[9];
layout(location=0) in vec3 pos;
layout(location=1) in vec3 nrm;
layout(location=2) in vec2 uv0;
layout(location=3) in vec4 tan0;
out vec3 v_wpos;
out vec3 v_nrm;
out vec3 v_tan;
out vec2 v_uv;
out float v_emis;
out float v_alpha;
void main() {
    mat4 mvp = mat4(vs_params[0], vs_params[1], vs_params[2], vs_params[3]);
    mat4 model = mat4(vs_params[4], vs_params[5], vs_params[6], vs_params[7]);
    v_wpos = (model * vec4(pos, 1.0)).xyz;
    v_nrm = mat3(model) * nrm;
    v_tan = mat3(model) * tan0.xyz;
    v_uv = uv0;
    v_emis = vs_params[8].x;
    v_alpha = vs_params[8].y;
    gl_Position = mvp * vec4(pos, 1.0);
}
)";

// frame[] layout:
//  0: camera pos, time          1: horizon (fog) rgb, fog density
//  2: left headlight pos, cos   3: right headlight pos, reach
//  4: beam dir, intensity       5: light count, streetlight level, night, -
//  6: key light dir, -          7: key light rgb, -
//  8: ambient sky rgb, -        9: ambient ground rgb, -
// 10 + 2i: light pos, range    11 + 2i: light rgb
// 42: zenith rgb, exposure     43: sun dir, sun visible
// mat[] layout:
//  0: emissive rgb, alpha cutoff (< 0 = opaque, -2 = blended)   1: base color
//  2: has detail, detail uv scale, has normal map, has maps texture
//  3: ksDiffuse, ksAmbient, ksSpecular, ksSpecularEXP
//  4: fresnelC, fresnelEXP, fresnelMaxLevel, isAdditive
// shadow[] layout: 0-3 near cascade matrix, 4-7 far cascade matrix,
//  8: near texel (world m), far texel (world m), shadow map texel (uv), enabled
const char* kSceneFS = R"(#version 410
uniform vec4 frame[44];
uniform vec4 mat[5];
uniform vec4 shadow[9];
uniform sampler2D tex;
uniform sampler2D detail_tex;
uniform sampler2D normal_tex;
uniform sampler2D maps_tex;
uniform sampler2DShadow shadow0;
uniform sampler2DShadow shadow1;
in vec3 v_wpos;
in vec3 v_nrm;
in vec3 v_tan;
in vec2 v_uv;
in float v_emis;
in float v_alpha;
out vec4 frag_color;

float beam(vec3 P, vec3 hp, vec3 dir, float cosOuter, float reach) {
    vec3 L = hp - P;
    float d = length(L);
    float c = dot(-L / d, dir);
    float cone = smoothstep(cosOuter, mix(cosOuter, 1.0, 0.45), c);
    return cone / (1.0 + (d * d) / (reach * reach));
}

// 3x3 PCF on a hardware-compared shadow map; returns 1 outside the map.
float pcf(sampler2DShadow s, mat4 m, vec3 p, out bool inside) {
    vec4 c = m * vec4(p, 1.0);
    vec3 q = c.xyz / c.w * 0.5 + 0.5;
    inside = all(greaterThan(q.xy, vec2(0.01))) && all(lessThan(q.xy, vec2(0.99))) && q.z < 1.0;
    if (!inside) return 1.0;
    float t = shadow[8].z, sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) sum += texture(s, vec3(q.xy + vec2(x, y) * t, q.z));
    return sum / 9.0;
}

float sunShadow(vec3 N, vec3 L) {
    if (shadow[8].w < 0.5) return 1.0;
    float slope = 1.0 - max(dot(N, L), 0.0);
    bool inside;
    // Normal-offset sampling avoids acne without detaching shadows from their casters.
    float s = pcf(shadow0, mat4(shadow[0], shadow[1], shadow[2], shadow[3]), v_wpos + N * shadow[8].x * (1.0 + 2.0 * slope), inside);
    if (inside) return s;
    s = pcf(shadow1, mat4(shadow[4], shadow[5], shadow[6], shadow[7]), v_wpos + N * shadow[8].y * (1.0 + 2.0 * slope), inside);
    return s;
}

vec3 skyColor(vec3 R) {
    vec3 horizon = frame[1].rgb, zenith = frame[42].rgb;
    vec3 c = mix(horizon, zenith, pow(clamp(R.y, 0.0, 1.0), 0.45));
    c = mix(c, horizon * 0.25, smoothstep(0.0, -0.3, R.y));  // ground below the horizon
    c += frame[7].rgb * pow(max(dot(R, frame[43].xyz), 0.0), 300.0) * 6.0 * frame[43].w;  // sun glint
    return c;
}

void main() {
    vec4 texel = texture(tex, v_uv) * mat[1];
    if (mat[2].x > 0.5) {
        // Assetto Corsa multimap: diffuse alpha 0 shows the detail (paint) texture.
        vec3 det = texture(detail_tex, v_uv * mat[2].y).rgb;
        texel = vec4(mix(det, texel.rgb, texel.a), 1.0);
    }
    float cutoff = mat[0].w;

    vec3 cam = frame[0].xyz;
    vec3 horizon = frame[1].rgb;
    float fogDen = frame[1].w;
    vec3 hl = frame[2].xyz, hr = frame[3].xyz;
    float cosOuter = frame[2].w, reach = frame[3].w;
    vec3 bdir = frame[4].xyz;
    float bint = frame[4].w;
    int nLights = int(frame[5].x);
    float lampLevel = frame[5].y;
    float night = frame[5].z;
    vec3 keyDir = frame[6].xyz, keyCol = frame[7].rgb;

    vec3 V = cam - v_wpos;
    float dist = length(V);
    V /= dist;
    vec3 Ng = normalize(v_nrm);
    if (!gl_FrontFacing) Ng = -Ng;
    vec3 N = Ng;
    if (mat[2].z > 0.5 && dot(v_tan, v_tan) > 0.25) {
        vec3 T = normalize(v_tan - Ng * dot(Ng, v_tan));
        vec3 B = cross(Ng, T);
        vec3 nm = texture(normal_tex, v_uv).xyz * 2.0 - 1.0;
        N = normalize(T * nm.x - B * nm.y + Ng * max(nm.z, 0.2));  // Direct3D-style green channel
    }
    vec4 maps = mat[2].w > 0.5 ? texture(maps_tex, v_uv) : vec4(1.0);
    float kd = mat[3].x * 2.5, ka = mat[3].y * 2.5;
    float ks = mat[3].z * maps.r, kexp = max(mat[3].w * (mat[2].w > 0.5 ? maps.g : 1.0), 1.0);
    vec3 albedo = pow(texel.rgb, vec3(2.2));

    float sh = sunShadow(Ng, keyDir);
    vec3 diffuse = mix(frame[9].rgb, frame[8].rgb, 0.5 + 0.5 * N.y) * ka * (0.7 + 0.3 * sh);
    vec3 spec = vec3(0.0);
    diffuse += keyCol * max(dot(N, keyDir), 0.0) * sh * kd;
    spec += keyCol * pow(max(dot(N, normalize(keyDir + V)), 0.0), kexp) * ks * sh;

    // Halogen headlights: warm white.
    vec3 hcol = vec3(1.0, 0.92, 0.78) * bint;
    for (int k = 0; k < 2; ++k) {
        vec3 hp = k == 0 ? hl : hr;
        vec3 L = normalize(hp - v_wpos);
        float a = beam(v_wpos, hp, bdir, cosOuter, reach);
        diffuse += hcol * a * max(dot(N, L), 0.0) * kd;
        spec += hcol * a * pow(max(dot(N, normalize(L + V)), 0.0), kexp) * (ks + 0.05);
    }
    // Track lights (sodium streetlights, signs, ...), switched on at dusk.
    for (int i = 0; i < nLights; ++i) {
        vec4 lp = frame[10 + 2 * i];
        vec3 lc = frame[11 + 2 * i].rgb * lampLevel;
        vec3 L = lp.xyz - v_wpos;
        float d = length(L);
        L /= d;
        float a = 1.0 / (1.0 + d * d / 30.0) * smoothstep(lp.w, lp.w * 0.6, d);
        diffuse += lc * a * max(dot(N, L), 0.0) * kd;
        spec += lc * a * pow(max(dot(N, normalize(L + V)), 0.0), kexp) * (ks + 0.05);
    }

    vec3 color = albedo * diffuse + spec + albedo * mat[0].rgb * v_emis * 4.0;

    // Reflections of the sky (paint, glass, chrome): Schlick-style fresnel as in AC.
    if (mat[4].z > 0.0) {
        float f = mat[4].x + (1.0 - mat[4].x) * pow(1.0 - max(dot(N, V), 0.0), mat[4].y);
        // Reflection replaces part of the base color instead of adding on top, so
        // painted bodies keep their color in bright daylight.
        float r = clamp(min(f, mat[4].z) * maps.b * 0.6, 0.0, 1.0);
        vec3 env = skyColor(reflect(-V, N));
        if (mat[4].w > 1.5) env *= texel.rgb / max(max(texel.r, max(texel.g, texel.b)), 0.05);  // metallic: tinted
        color = mix(color, env, r);
    }

    // Fog takes the sky's horizon color (brighter toward the sun), plus light
    // scattered toward the camera by the fog at night.
    float fog = exp(-pow(fogDen * dist, 2.0));
    vec3 fogCol = horizon + keyCol * 0.08 * pow(max(dot(-V, keyDir), 0.0), 6.0);
    vec3 scatter = vec3(0.0);
    for (int i = 0; i < nLights; ++i) {
        vec4 lp = frame[10 + 2 * i];
        vec3 lc = frame[11 + 2 * i].rgb * lampLevel;
        float t = clamp(dot(lp.xyz - cam, -V), 0.0, dist);
        float d = length(cam - V * t - lp.xyz);
        scatter += lc * 0.05 / (1.0 + d * d * 0.35);
    }
    float stepLen = min(dist, 40.0) / 8.0;
    for (int s = 0; s < 8; ++s) {
        vec3 P = cam - V * (stepLen * (float(s) + 0.5));
        scatter += hcol * (beam(P, hl, bdir, cosOuter, reach) + beam(P, hr, bdir, cosOuter, reach)) * stepLen * 0.004;
    }
    color = mix(fogCol, color, fog) + scatter * 0.34 * (0.15 + 0.85 * night);

    color = vec3(1.0) - exp(-color * frame[42].w);  // exposure + soft highlight shoulder
    float alpha = 1.0;
    if (cutoff < -1.5) alpha = texel.a;  // blended
    else if (cutoff >= 0.0) alpha = clamp((texel.a - cutoff) / max(fwidth(texel.a), 1e-4) + 0.5, 0.0, 1.0);  // alpha to coverage
    frag_color = vec4(pow(color, vec3(1.0 / 2.2)), alpha * v_alpha);
}
)";

// Shadow map pass: depth only; alpha-tested materials cut out their texture.
const char* kShadowVS = R"(#version 410
uniform vec4 sh_vs[4];
layout(location=0) in vec3 pos;
layout(location=1) in vec2 uv0;
out vec2 v_uv;
void main() {
    v_uv = uv0;
    gl_Position = mat4(sh_vs[0], sh_vs[1], sh_vs[2], sh_vs[3]) * vec4(pos, 1.0);
}
)";

const char* kShadowFS = R"(#version 410
uniform vec4 sh_fs[1];
uniform sampler2D tex;
in vec2 v_uv;
void main() {
    if (sh_fs[0].x >= 0.0 && texture(tex, v_uv).a < sh_fs[0].x) discard;
}
)";

// Sky: gradient, sun, moon, stars and a thin cloud layer, drawn behind everything.
// sky[] layout: 0 forward, time   1 right * tan(fovx/2)   2 up * tan(fovy/2)
//   3 sun dir, sun visible   4 sun rgb, night   5 zenith rgb, sunset   6 horizon rgb   7 moon dir
const char* kSkyVS = R"(#version 410
out vec2 ndc;
const vec2 corners[3] = vec2[3](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
void main() {
    ndc = corners[gl_VertexID];
    gl_Position = vec4(ndc, 1.0, 1.0);
}
)";

const char* kSkyFS = R"(#version 410
uniform vec4 sky[8];
in vec2 ndc;
out vec4 frag_color;

float hash3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float valueNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash3(vec3(i, 0)), b = hash3(vec3(i + vec2(1, 0), 0));
    float c = hash3(vec3(i + vec2(0, 1), 0)), d = hash3(vec3(i + vec2(1, 1), 0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int k = 0; k < 5; ++k) { s += a * valueNoise(p); p *= 2.07; a *= 0.5; }
    return s;
}

void main() {
    vec3 dir = normalize(sky[0].xyz + sky[1].xyz * ndc.x + sky[2].xyz * ndc.y);
    float t = sky[0].w;
    vec3 sunDir = sky[3].xyz, sunCol = sky[4].rgb, zenith = sky[5].rgb, horizon = sky[6].rgb, moonDir = sky[7].xyz;
    float sunVis = sky[3].w, night = sky[4].w, sunset = sky[5].w;

    float h = dir.y;
    vec3 col = mix(horizon, zenith, pow(clamp(h, 0.0, 1.0), 0.45));
    col = mix(col, horizon * 0.35, smoothstep(0.0, -0.25, h));  // below the horizon: darker haze

    float sd = max(dot(dir, sunDir), 0.0);
    col += sunCol * (pow(sd, 8.0) * 0.25 + pow(sd, 64.0) * 0.6) * (sunVis + sunset);
    col += sunCol * smoothstep(0.9993, 0.9996, sd) * 20.0 * sunVis;

    // Stars and moon at night.
    if (night > 0.0 && h > 0.0) {
        vec3 cell = floor(dir * 300.0);
        float star = step(0.9975, hash3(cell)) * hash3(cell + 7.0);
        float twinkle = 0.7 + 0.3 * sin(t * 3.0 + hash3(cell) * 50.0);
        col += vec3(0.8, 0.85, 1.0) * star * twinkle * night * smoothstep(0.0, 0.2, h) * 0.6;
        float md = dot(dir, moonDir);
        col += vec3(0.75, 0.8, 0.9) * smoothstep(0.99955, 0.9997, md) * night * 1.5;
        col += vec3(0.25, 0.3, 0.4) * pow(max(md, 0.0), 40.0) * night * 0.08;
    }

    // Thin, slowly drifting cloud layer.
    if (h > 0.0) {
        vec2 uv = dir.xz / (h + 0.12) * 1.3 + vec2(t * 0.004, t * 0.002);
        float c = smoothstep(0.5, 0.85, fbm(uv));
        vec3 lit = mix(horizon * 0.6, sunCol * (0.6 + 0.6 * sd), clamp(sunVis + sunset, 0.0, 1.0));
        vec3 cloudCol = mix(lit, vec3(0.012, 0.014, 0.02), night);
        col = mix(col, cloudCol, c * 0.75 * smoothstep(0.0, 0.15, h));
    }

    col = vec3(1.0) - exp(-col * sky[6].w);
    frag_color = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);
}
)";

const char* kOsdVS = R"(#version 410
uniform vec4 osd_vs[1];
layout(location=0) in vec2 pos;
layout(location=1) in vec2 uv0;
layout(location=2) in vec4 col0;
out vec2 uv;
out vec4 col;
void main() {
    uv = uv0;
    col = col0;
    gl_Position = vec4((pos.x / 320.0 - 1.0) * osd_vs[0].x, (1.0 - pos.y / 240.0) * osd_vs[0].y, 0.0, 1.0);
}
)";

const char* kOsdFS = R"(#version 410
uniform sampler2D tex;
in vec2 uv;
in vec4 col;
out vec4 frag_color;
void main() {
    frag_color = vec4(col.rgb, col.a * texture(tex, uv).a);
}
)";

const char* kPostVS = R"(#version 410
uniform vec4 post_vs[1];
out vec2 uv;
const vec2 corners[6] = vec2[6](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,-1), vec2(1,1), vec2(-1,1));
void main() {
    vec2 p = corners[gl_VertexID];
    uv = p * 0.5 + 0.5;
    gl_Position = vec4(p * post_vs[0].xy, 0.0, 1.0);
}
)";

const char* kPostFS = R"(#version 410
uniform vec4 post_fs[1];
uniform sampler2D tex;
in vec2 uv;
out vec4 frag_color;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec3 toYIQ(vec3 c) {
    return vec3(dot(c, vec3(0.299, 0.587, 0.114)), dot(c, vec3(0.596, -0.274, -0.322)), dot(c, vec3(0.211, -0.523, 0.312)));
}
vec3 toRGB(vec3 y) {
    return vec3(y.x + 0.956 * y.y + 0.621 * y.z, y.x - 0.272 * y.y - 0.647 * y.z, y.x - 1.106 * y.y + 1.703 * y.z);
}

void main() {
    float t = post_fs[0].x;
    vec2 px = 1.0 / vec2(640.0, 480.0);
    float line = floor(uv.y * 480.0);

    float vhs = post_fs[0].y;  // 0 = clean picture, 1 = tape look
    // Time-base wobble: faint per-line jitter plus head-switching noise confined to
    // the bottom edge (no rolling bands or dropouts across the picture).
    float wob = (hash(vec2(line, floor(t * 30.0))) - 0.5) * 0.35 * px.x;
    float bottom = smoothstep(0.02, 0.0, uv.y);
    wob += (sin(uv.y * 300.0 + t * 17.0) * 3.0 + hash(vec2(line, t)) * 4.0) * px.x * bottom;
    vec2 u = uv + vec2(wob * vhs, 0.0);

    // Luma is only slightly soft; chroma is smeared wide and shifted right.
    float Y = 0.0;
    const float lw[5] = float[5](0.1, 0.2, 0.4, 0.2, 0.1);
    for (int k = -2; k <= 2; ++k) Y += toYIQ(texture(tex, u + vec2(float(k) * px.x, 0.0)).rgb).x * lw[k + 2];
    vec2 iq = vec2(0.0);
    for (int k = -4; k <= 4; ++k) iq += toYIQ(texture(tex, u + vec2((float(k) * 1.6 + 2.0) * px.x, 0.0)).rgb).yz;
    iq /= 9.0;
    vec3 col = toRGB(vec3(Y, iq * 1.15));

    // Highlights bloom and bleed, like a CCD at night.
    vec3 bloom = vec3(0.0);
    for (int k = 0; k < 8; ++k) {
        float a = float(k) * 0.785398;
        vec2 d = vec2(cos(a), sin(a)) * px;
        bloom += max(texture(tex, u + d * 3.0).rgb - 0.68, 0.0);
        bloom += max(texture(tex, u + d * 8.0).rgb - 0.68, 0.0) * 0.6;
    }
    col += bloom * 0.09;
    // Horizontal streak from very bright sources.
    vec3 streak = vec3(0.0);
    for (int k = 1; k <= 6; ++k) {
        streak += max(texture(tex, u + vec2(float(k) * 4.0 * px.x, 0.0)).rgb - 0.7, 0.0);
        streak += max(texture(tex, u - vec2(float(k) * 4.0 * px.x, 0.0)).rgb - 0.7, 0.0);
    }
    col += streak * 0.05;

    // Grain (stronger in the dark, where AGC pumps the gain) and dropouts.
    float lum = dot(col, vec3(0.3, 0.59, 0.11));
    float grain = hash(floor(uv * post_fs[0].zw * 0.5) + fract(t * 7.13) * 91.0) - 0.5;
    col += grain * mix(0.09, 0.03, clamp(lum * 2.0, 0.0, 1.0));
    vec2 cn = vec2(hash(floor(uv * vec2(160.0, 240.0)) + t), hash(floor(uv * vec2(160.0, 240.0)) - t)) - 0.5;
    col += toRGB(vec3(0.0, cn * 0.03));

    col *= 0.96 + 0.04 * sin(uv.y * 480.0 * 3.14159);
    col = col * 0.94 + vec3(0.015, 0.018, 0.022);  // lifted, slightly green-blue blacks

    // Clean path: the plain picture with the same bloom.
    vec3 clean = texture(tex, uv).rgb + bloom * 0.07;
    col = mix(clean, col, vhs);
    vec2 v = uv - 0.5;
    col *= 1.0 - dot(v, v) * mix(0.35, 0.7, vhs);
    frag_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
)";

// ---------------------------------------------------------------------------
// 5x7 bitmap font for the camcorder OSD. Each glyph is 7 rows of 5 bits.

struct Glyph {
    char c;
    uint8_t rows[7];
};
const Glyph kFont[] = {
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}}, {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}}, {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}}, {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}}, {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}}, {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {'A', {0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11}}, {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}},
    {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}}, {'D', {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C}},
    {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}}, {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}},
    {'G', {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}}, {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}}, {'J', {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C}},
    {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}}, {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}},
    {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}}, {'N', {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11}},
    {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}}, {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
    {'Q', {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}}, {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
    {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}}, {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
    {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}}, {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}},
    {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}}, {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}},
    {'Y', {0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04}}, {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
    {':', {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}}, {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}},
    {'/', {0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00}}, {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}},
    {'+', {0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00}},
    {'[', {0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E}}, {']', {0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E}},
    {'(', {0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02}}, {')', {0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08}},
    {'*', {0x00, 0x0E, 0x1F, 0x1F, 0x1F, 0x0E, 0x00}},  // record dot
    {'>', {0x10, 0x18, 0x1C, 0x1E, 0x1C, 0x18, 0x10}},  // play triangle
};
constexpr int kCellW = 6, kCellH = 8, kAtlasCols = 16, kAtlasW = kCellW * kAtlasCols, kAtlasH = kCellH * 8;
constexpr int kSolidGlyph = 127;

void glyphUV(int code, float& u0, float& v0, float& u1, float& v1) {
    int cx = (code % kAtlasCols) * kCellW, cy = (code / kAtlasCols) * kCellH;
    u0 = float(cx) / kAtlasW;
    v0 = float(cy) / kAtlasH;
    u1 = float(cx + 5) / kAtlasW;
    v1 = float(cy + 7) / kAtlasH;
}

void solidUV(float& u, float& v) {
    u = (float((kSolidGlyph % kAtlasCols) * kCellW) + 3.0f) / kAtlasW;
    v = (float((kSolidGlyph / kAtlasCols) * kCellH) + 4.0f) / kAtlasH;
}

struct GpuMaterial {
    sg_view tex{}, detail{}, normal{}, maps{};
    float params[kMatVec4][4];  // see kSceneFS mat[] layout
    int pass = 0;               // 0 opaque/cull, 1 opaque/two-sided, 2 blended, 3 alpha-tested
};

// Draw order of the passes: opaque, two-sided, alpha-tested, then blended.
int passOrder(int pass) {
    static const int order[4] = {0, 1, 3, 2};
    return order[pass];
}

struct GpuModel {
    sg_buffer pos{}, nrm{}, uv{}, tan{}, idx{};
    std::vector<sg_image> images;
    std::vector<sg_view> views;
    std::vector<GpuMaterial> materials;
    std::vector<MeshGroup> groups;
};

struct DrawItem {
    const GpuModel* model;
    const Primitive* prim;
    const GpuMaterial* mat;
    M4 xf;
    float emissive;
    float alpha;
    int pass;  // pipeline: the material's, or blended for see-through draws
    float depth;
};

sg_buffer makeBuffer(const void* data, size_t size, bool index, const char* label) {
    sg_buffer_desc d = {};
    d.usage.vertex_buffer = !index;
    d.usage.index_buffer = index;
    d.data = {data, size};
    d.label = label;
    return sg_make_buffer(&d);
}

}  // namespace

// ---------------------------------------------------------------------------

struct Renderer::Gpu {
    sg_shader sceneShd{}, osdShd{}, postShd{}, skyShd{}, shadowShd{};
    sg_pipeline scenePip[4]{}, osdPip{}, postPip{}, skyPip{}, shadowPip{};
    // Sun/moon shadow cascades (near, far).
    sg_image shadowImg[2]{};
    sg_view shadowAtt[2]{}, shadowTex[2]{};
    sg_sampler shadowSmp{};
    // Scene target: MSAA color + depth, resolved into a texture for the post pass.
    static constexpr int kMsaa = 4;
    int targetW = 0, targetH = 0;
    sg_image msaaImg{}, resolveImg{}, depthImg{}, fontImg{}, whiteImg{};
    sg_view colorAtt{}, resolveAtt{}, depthAtt{}, colorTex{}, fontTex{}, whiteTex{};

    void ensureTarget(int w, int h) {
        if (w == targetW && h == targetH) return;
        for (sg_view v : {colorAtt, resolveAtt, depthAtt, colorTex})
            if (v.id) sg_destroy_view(v);
        for (sg_image i : {msaaImg, resolveImg, depthImg})
            if (i.id) sg_destroy_image(i);
        targetW = w;
        targetH = h;
        sg_image_desc d = {};
        d.width = w;
        d.height = h;
        d.pixel_format = SG_PIXELFORMAT_RGBA8;
        d.sample_count = kMsaa;
        d.usage.color_attachment = true;
        d.label = "scene-msaa";
        msaaImg = sg_make_image(&d);
        d.sample_count = 1;
        d.usage = {};
        d.usage.resolve_attachment = true;
        d.label = "scene-resolve";
        resolveImg = sg_make_image(&d);
        d.sample_count = kMsaa;
        d.usage = {};
        d.usage.depth_stencil_attachment = true;
        d.pixel_format = SG_PIXELFORMAT_DEPTH_STENCIL;
        d.label = "scene-depth";
        depthImg = sg_make_image(&d);
        sg_view_desc v = {};
        v.color_attachment.image = msaaImg;
        colorAtt = sg_make_view(&v);
        v = {};
        v.resolve_attachment.image = resolveImg;
        resolveAtt = sg_make_view(&v);
        v = {};
        v.depth_stencil_attachment.image = depthImg;
        depthAtt = sg_make_view(&v);
        v = {};
        v.texture.image = resolveImg;
        colorTex = sg_make_view(&v);
    }
    sg_sampler linearSmp{}, nearestSmp{}, materialSmp{};
    sg_buffer osdBuf{};
    std::vector<std::unique_ptr<GpuModel>> models;
    std::vector<DrawItem> queue;
    static constexpr int kOsdMaxVerts = 1 << 16;
};

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

bool Renderer::init() {
    sg_desc desc = {};
    desc.environment.defaults.color_format = SG_PIXELFORMAT_RGBA8;
    desc.environment.defaults.depth_format = SG_PIXELFORMAT_DEPTH_STENCIL;
    desc.environment.defaults.sample_count = 1;
    desc.image_pool_size = 2048;  // AC tracks easily have hundreds of textures
    desc.view_pool_size = 2048;
    desc.buffer_pool_size = 256;
    desc.logger.func = slog_func;
    sg_setup(&desc);
    if (!sg_isvalid()) return false;

    gpu = std::make_unique<Gpu>();
    Gpu& g = *gpu;

    // Scene shader and pipelines.
    {
        sg_shader_desc sd = {};
        sd.vertex_func.source = kSceneVS;
        sd.fragment_func.source = kSceneFS;
        const char* names[4] = {"pos", "nrm", "uv0", "tan0"};
        for (int k = 0; k < 4; ++k) {
            sd.attrs[k].glsl_name = names[k];
            sd.attrs[k].base_type = SG_SHADERATTRBASETYPE_FLOAT;
        }
        sd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
        sd.uniform_blocks[0].size = 9 * 16;
        sd.uniform_blocks[0].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 9, "vs_params"};
        sd.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.uniform_blocks[1].size = kFrameVec4 * 16;
        sd.uniform_blocks[1].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[1].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, kFrameVec4, "frame"};
        sd.uniform_blocks[2].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.uniform_blocks[2].size = kMatVec4 * 16;
        sd.uniform_blocks[2].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[2].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, kMatVec4, "mat"};
        sd.uniform_blocks[3].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.uniform_blocks[3].size = kShadowVec4 * 16;
        sd.uniform_blocks[3].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[3].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, kShadowVec4, "shadow"};
        for (int v = 0; v < 6; ++v) {
            sd.views[v].texture.stage = SG_SHADERSTAGE_FRAGMENT;
            sd.views[v].texture.image_type = SG_IMAGETYPE_2D;
            sd.views[v].texture.sample_type = v < 4 ? SG_IMAGESAMPLETYPE_FLOAT : SG_IMAGESAMPLETYPE_DEPTH;
        }
        sd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        sd.samplers[1].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.samplers[1].sampler_type = SG_SAMPLERTYPE_COMPARISON;
        const char* texNames[6] = {"tex", "detail_tex", "normal_tex", "maps_tex", "shadow0", "shadow1"};
        for (int v = 0; v < 6; ++v)
            sd.texture_sampler_pairs[v] = {SG_SHADERSTAGE_FRAGMENT, uint8_t(v), uint8_t(v < 4 ? 0 : 1), texNames[v]};
        sd.label = "scene-shader";
        g.sceneShd = sg_make_shader(&sd);

        for (int pass = 0; pass < 4; ++pass) {
            sg_pipeline_desc pd = {};
            pd.shader = g.sceneShd;
            pd.layout.attrs[0] = {0, 0, SG_VERTEXFORMAT_FLOAT3};
            pd.layout.attrs[1] = {1, 0, SG_VERTEXFORMAT_FLOAT3};
            pd.layout.attrs[2] = {2, 0, SG_VERTEXFORMAT_FLOAT2};
            pd.layout.attrs[3] = {3, 0, SG_VERTEXFORMAT_FLOAT4};
            pd.alpha_to_coverage_enabled = pass == 3;  // smooth foliage edges under MSAA
            pd.index_type = SG_INDEXTYPE_UINT32;
            pd.cull_mode = pass == 0 ? SG_CULLMODE_BACK : SG_CULLMODE_NONE;
            // Converted AC meshes are counter-clockwise in our right-handed space.
            pd.face_winding = SG_FACEWINDING_CCW;
            pd.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
            pd.depth.write_enabled = pass != 2;
            if (pass == 2) {
                pd.colors[0].blend.enabled = true;
                pd.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA;
                pd.colors[0].blend.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
            }
            pd.sample_count = Gpu::kMsaa;
            pd.label = "scene-pipeline";
            g.scenePip[pass] = sg_make_pipeline(&pd);
        }
    }

    // Shadow map shader, pipeline and cascade images.
    {
        sg_shader_desc sd = {};
        sd.vertex_func.source = kShadowVS;
        sd.fragment_func.source = kShadowFS;
        sd.attrs[0].glsl_name = "pos";
        sd.attrs[1].glsl_name = "uv0";
        sd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
        sd.uniform_blocks[0].size = 4 * 16;
        sd.uniform_blocks[0].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 4, "sh_vs"};
        sd.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.uniform_blocks[1].size = 16;
        sd.uniform_blocks[1].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[1].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 1, "sh_fs"};
        sd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        sd.views[0].texture.image_type = SG_IMAGETYPE_2D;
        sd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        sd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        sd.texture_sampler_pairs[0] = {SG_SHADERSTAGE_FRAGMENT, 0, 0, "tex"};
        sd.label = "shadow-shader";
        g.shadowShd = sg_make_shader(&sd);

        sg_pipeline_desc pd = {};
        pd.shader = g.shadowShd;
        pd.layout.attrs[0] = {0, 0, SG_VERTEXFORMAT_FLOAT3};
        pd.layout.attrs[1] = {1, 0, SG_VERTEXFORMAT_FLOAT2};
        pd.index_type = SG_INDEXTYPE_UINT32;
        pd.cull_mode = SG_CULLMODE_NONE;
        pd.color_count = 0;
        pd.depth.pixel_format = SG_PIXELFORMAT_DEPTH;
        pd.depth.compare = SG_COMPAREFUNC_LESS_EQUAL;
        pd.depth.write_enabled = true;
        pd.depth.bias = 1.0f;
        pd.depth.bias_slope_scale = 2.0f;
        pd.sample_count = 1;
        pd.label = "shadow-pipeline";
        g.shadowPip = sg_make_pipeline(&pd);

        for (int c = 0; c < 2; ++c) {
            sg_image_desc d = {};
            d.usage.depth_stencil_attachment = true;
            d.width = kShadowSize;
            d.height = kShadowSize;
            d.pixel_format = SG_PIXELFORMAT_DEPTH;
            d.sample_count = 1;
            d.label = "shadow-cascade";
            g.shadowImg[c] = sg_make_image(&d);
            sg_view_desc v = {};
            v.depth_stencil_attachment.image = g.shadowImg[c];
            g.shadowAtt[c] = sg_make_view(&v);
            v = {};
            v.texture.image = g.shadowImg[c];
            g.shadowTex[c] = sg_make_view(&v);
        }
        sg_sampler_desc sm = {};
        sm.min_filter = SG_FILTER_LINEAR;
        sm.mag_filter = SG_FILTER_LINEAR;
        sm.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
        sm.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
        sm.compare = SG_COMPAREFUNC_LESS_EQUAL;
        g.shadowSmp = sg_make_sampler(&sm);
    }

    // Sky shader and pipeline: fullscreen, behind everything.
    {
        sg_shader_desc sd = {};
        sd.vertex_func.source = kSkyVS;
        sd.fragment_func.source = kSkyFS;
        sd.uniform_blocks[0].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.uniform_blocks[0].size = kSkyVec4 * 16;
        sd.uniform_blocks[0].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, kSkyVec4, "sky"};
        sd.label = "sky-shader";
        g.skyShd = sg_make_shader(&sd);
        sg_pipeline_desc pd = {};
        pd.shader = g.skyShd;
        pd.depth.compare = SG_COMPAREFUNC_ALWAYS;
        pd.depth.write_enabled = false;
        pd.sample_count = Gpu::kMsaa;
        pd.label = "sky-pipeline";
        g.skyPip = sg_make_pipeline(&pd);
    }

    // OSD shader and pipeline (alpha blended, no depth).
    {
        sg_shader_desc sd = {};
        sd.vertex_func.source = kOsdVS;
        sd.fragment_func.source = kOsdFS;
        sd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
        sd.uniform_blocks[0].size = 16;
        sd.uniform_blocks[0].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 1, "osd_vs"};
        const char* names[3] = {"pos", "uv0", "col0"};
        for (int k = 0; k < 3; ++k) {
            sd.attrs[k].glsl_name = names[k];
            sd.attrs[k].base_type = SG_SHADERATTRBASETYPE_FLOAT;
        }
        sd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        sd.views[0].texture.image_type = SG_IMAGETYPE_2D;
        sd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        sd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        sd.texture_sampler_pairs[0] = {SG_SHADERSTAGE_FRAGMENT, 0, 0, "tex"};
        sd.label = "osd-shader";
        g.osdShd = sg_make_shader(&sd);

        sg_pipeline_desc pd = {};
        pd.shader = g.osdShd;
        pd.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT2;
        pd.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT2;
        pd.layout.attrs[2].format = SG_VERTEXFORMAT_UBYTE4N;
        pd.colors[0].blend.enabled = true;
        pd.colors[0].blend.src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA;
        pd.colors[0].blend.dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
        pd.depth.compare = SG_COMPAREFUNC_ALWAYS;
        pd.depth.write_enabled = false;
        pd.sample_count = Gpu::kMsaa;
        pd.label = "osd-pipeline";
        g.osdPip = sg_make_pipeline(&pd);

        sg_buffer_desc bd = {};
        bd.usage.vertex_buffer = true;
        bd.usage.write_transient = true;
        bd.size = Gpu::kOsdMaxVerts * sizeof(Osd::Vert);
        bd.label = "osd-verts";
        g.osdBuf = sg_make_buffer(&bd);
    }

    // Post shader and pipeline.
    {
        sg_shader_desc sd = {};
        sd.vertex_func.source = kPostVS;
        sd.fragment_func.source = kPostFS;
        sd.uniform_blocks[0].stage = SG_SHADERSTAGE_VERTEX;
        sd.uniform_blocks[0].size = 16;
        sd.uniform_blocks[0].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[0].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 1, "post_vs"};
        sd.uniform_blocks[1].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.uniform_blocks[1].size = 16;
        sd.uniform_blocks[1].layout = SG_UNIFORMLAYOUT_STD140;
        sd.uniform_blocks[1].glsl_uniforms[0] = {SG_UNIFORMTYPE_FLOAT4, 1, "post_fs"};
        sd.views[0].texture.stage = SG_SHADERSTAGE_FRAGMENT;
        sd.views[0].texture.image_type = SG_IMAGETYPE_2D;
        sd.views[0].texture.sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        sd.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        sd.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        sd.texture_sampler_pairs[0] = {SG_SHADERSTAGE_FRAGMENT, 0, 0, "tex"};
        sd.label = "post-shader";
        g.postShd = sg_make_shader(&sd);

        sg_pipeline_desc pd = {};
        pd.shader = g.postShd;
        pd.label = "post-pipeline";
        g.postPip = sg_make_pipeline(&pd);
    }

    // Font atlas and a 1x1 white texture for untextured materials.
    {
        std::vector<uint32_t> px(kAtlasW * kAtlasH, 0x00FFFFFFu);
        auto put = [&](int code, const uint8_t rows[7]) {
            int cx = (code % kAtlasCols) * kCellW, cy = (code / kAtlasCols) * kCellH;
            for (int r = 0; r < 7; ++r)
                for (int c = 0; c < 5; ++c)
                    if (rows[r] & (0x10 >> c)) px[(cy + r) * kAtlasW + cx + c] = 0xFFFFFFFFu;
        };
        for (const Glyph& gl : kFont) put(gl.c, gl.rows);
        int sx = (kSolidGlyph % kAtlasCols) * kCellW, sy = (kSolidGlyph / kAtlasCols) * kCellH;
        for (int r = 0; r < kCellH; ++r)
            for (int c = 0; c < kCellW; ++c) px[(sy + r) * kAtlasW + sx + c] = 0xFFFFFFFFu;
        sg_image_desc fd = {};
        fd.width = kAtlasW;
        fd.height = kAtlasH;
        fd.pixel_format = SG_PIXELFORMAT_RGBA8;
        fd.data.mip_levels[0] = {px.data(), px.size() * 4};
        fd.label = "osd-font";
        g.fontImg = sg_make_image(&fd);
        sg_view_desc v = {};
        v.texture.image = g.fontImg;
        g.fontTex = sg_make_view(&v);

        static const uint32_t white = 0xFFFFFFFFu;
        sg_image_desc wd = {};
        wd.width = 1;
        wd.height = 1;
        wd.pixel_format = SG_PIXELFORMAT_RGBA8;
        wd.data.mip_levels[0] = {&white, 4};
        g.whiteImg = sg_make_image(&wd);
        v = {};
        v.texture.image = g.whiteImg;
        g.whiteTex = sg_make_view(&v);
    }

    sg_sampler_desc sm = {};
    sm.min_filter = SG_FILTER_LINEAR;
    sm.mag_filter = SG_FILTER_LINEAR;
    sm.wrap_u = SG_WRAP_CLAMP_TO_EDGE;
    sm.wrap_v = SG_WRAP_CLAMP_TO_EDGE;
    g.linearSmp = sg_make_sampler(&sm);
    sm.min_filter = SG_FILTER_NEAREST;
    sm.mag_filter = SG_FILTER_NEAREST;
    g.nearestSmp = sg_make_sampler(&sm);
    sm = {};
    sm.min_filter = SG_FILTER_LINEAR;
    sm.mag_filter = SG_FILTER_LINEAR;
    sm.mipmap_filter = SG_FILTER_LINEAR;
    sm.wrap_u = SG_WRAP_REPEAT;
    sm.wrap_v = SG_WRAP_REPEAT;
    sm.max_anisotropy = 4;
    g.materialSmp = sg_make_sampler(&sm);
    return true;
}

void Renderer::shutdown() {
    gpu.reset();
    sg_shutdown();
}

int Renderer::addModel(const ModelData& m) {
    Gpu& g = *gpu;
    auto gm = std::make_unique<GpuModel>();
    gm->pos = makeBuffer(m.pos.data(), m.pos.size() * 4, false, "model-pos");
    gm->nrm = makeBuffer(m.nrm.data(), m.nrm.size() * 4, false, "model-nrm");
    gm->uv = makeBuffer(m.uv.data(), m.uv.size() * 4, false, "model-uv");
    gm->tan = makeBuffer(m.tan.data(), m.tan.size() * 4, false, "model-tan");
    gm->idx = makeBuffer(m.idx.data(), m.idx.size() * 4, true, "model-idx");

    for (const ImageData& im : m.images) {
        sg_view view = g.whiteTex;
        if (!im.mips.empty()) {
            sg_image_desc d = {};
            d.width = im.width;
            d.height = im.height;
            d.pixel_format = im.format == ImageData::BC1   ? SG_PIXELFORMAT_BC1_RGBA
                             : im.format == ImageData::BC2 ? SG_PIXELFORMAT_BC2_RGBA
                             : im.format == ImageData::BC3 ? SG_PIXELFORMAT_BC3_RGBA
                                                           : SG_PIXELFORMAT_RGBA8;
            d.num_mipmaps = std::min<int>(int(im.mips.size()), SG_MAX_MIPMAPS);
            for (int k = 0; k < d.num_mipmaps; ++k) d.data.mip_levels[k] = {im.mips[k].data(), im.mips[k].size()};
            sg_image img = sg_make_image(&d);
            gm->images.push_back(img);
            sg_view_desc vd = {};
            vd.texture.image = img;
            view = sg_make_view(&vd);
        }
        gm->views.push_back(view);
    }

    for (const MaterialData& md : m.materials) {
        GpuMaterial gmat;
        auto viewOf = [&](int image) { return (image >= 0 && image < int(gm->views.size())) ? gm->views[image] : g.whiteTex; };
        gmat.tex = viewOf(md.image);
        gmat.detail = viewOf(md.detailImage);
        gmat.normal = viewOf(md.normalImage);
        gmat.maps = viewOf(md.mapsImage);
        std::memset(gmat.params, 0, sizeof(gmat.params));
        gmat.params[2][0] = md.detailImage >= 0 ? 1.0f : 0.0f;
        gmat.params[2][1] = md.detailUv;
        gmat.params[2][2] = md.normalImage >= 0 ? 1.0f : 0.0f;
        gmat.params[2][3] = md.mapsImage >= 0 ? 1.0f : 0.0f;
        gmat.params[3][0] = md.ksDiffuse;
        gmat.params[3][1] = md.ksAmbient;
        gmat.params[3][2] = md.ksSpecular;
        gmat.params[3][3] = md.ksSpecularExp;
        gmat.params[4][0] = md.fresnelC;
        gmat.params[4][1] = md.fresnelExp;
        gmat.params[4][2] = md.fresnelMax;
        gmat.params[4][3] = md.isAdditive;
        gmat.params[0][0] = md.emissive[0];
        gmat.params[0][1] = md.emissive[1];
        gmat.params[0][2] = md.emissive[2];
        gmat.params[0][3] = md.alpha == MaterialData::Mask ? md.cutoff : (md.alpha == MaterialData::Blend ? -2.0f : -1.0f);
        std::memcpy(gmat.params[1], md.baseColor, sizeof(md.baseColor));
        gmat.pass = md.alpha == MaterialData::Blend ? 2 : md.alpha == MaterialData::Mask ? 3 : (md.doubleSided ? 1 : 0);
        gm->materials.push_back(gmat);
    }
    gm->groups = m.groups;
    g.models.push_back(std::move(gm));
    return int(g.models.size()) - 1;
}

void Renderer::submit(int model, int group, const M4& xf, float emissiveScale, float alpha) {
    if (model < 0 || model >= int(gpu->models.size())) return;
    const GpuModel* gm = gpu->models[model].get();
    if (group < 0 || group >= int(gm->groups.size())) return;
    static const GpuMaterial fallback = {{}, {}, {}, {}, {{0, 0, 0, -1}, {1, 1, 1, 1}, {0, 1, 0, 0}, {0.4f, 0.4f, 0, 20}, {0, 5, 0, 0}}, 0};
    for (const Primitive& p : gm->groups[group].prims) {
        const GpuMaterial* mat = (p.material >= 0 && p.material < int(gm->materials.size())) ? &gm->materials[p.material] : &fallback;
        gpu->queue.push_back({gm, &p, mat, xf, emissiveScale, alpha, alpha < 0.999f ? 2 : mat->pass, 0.0f});
    }
}

void Renderer::render(const RenderView& rv, int fbW, int fbH) {
    Gpu& g = *gpu;
    const M4 viewProj = rv.proj * rv.view;

    // Per-frame lighting uniforms.
    const Environment& env = rv.env;
    float frame[kFrameVec4][4] = {};
    auto set4 = [&](int k, V3 v, float w) { frame[k][0] = v.x; frame[k][1] = v.y; frame[k][2] = v.z; frame[k][3] = w; };
    set4(0, rv.camPos, rv.time);
    set4(1, env.horizon, env.fogDensity);
    set4(2, rv.headlightPos[0], std::cos(rv.highBeam ? 0.30f : 0.42f));
    set4(3, rv.headlightPos[1], rv.highBeam ? 40.0f : 18.0f);
    set4(4, rv.beamDir, rv.lightsOn ? (rv.highBeam ? 3.2f : 2.4f) : 0.0f);
    set4(6, env.keyDir, 0);
    set4(7, env.keyColor, 0);
    set4(8, env.ambientSky, 0);
    set4(9, env.ambientGround, 0);
    int nLights = 0;
    if (trackLights) {
        std::vector<std::pair<float, const TrackLight*>> near;
        near.reserve(trackLights->size());
        for (const TrackLight& l : *trackLights) {
            V3 d = l.pos - rv.camPos;
            near.push_back({dot(d, d), &l});
        }
        nLights = std::min<int>(kMaxLights, int(near.size()));
        std::partial_sort(near.begin(), near.begin() + nLights, near.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
        for (int k = 0; k < nLights; ++k) {
            const TrackLight& l = *near[k].second;
            float peak = std::max({l.color.x, l.color.y, l.color.z, 1e-4f});
            set4(10 + 2 * k, l.pos, std::max(l.range, 20.0f) * 1.6f);
            set4(11 + 2 * k, l.color * (2.4f / peak), 0);
        }
    }
    set4(5, {float(nLights), smoothstep(0.15f, 0.6f, env.night), env.night}, 0);
    set4(42, env.zenith, lerpf(1.9f, 1.45f, env.day));
    set4(43, env.sunDir, smoothstep(-0.03f, 0.02f, env.sunDir.y));

    // Sky uniforms: camera basis from the view matrix rows, frustum extents from the projection.
    float skyU[kSkyVec4][4] = {};
    auto sky4 = [&](int k, V3 v, float w) { skyU[k][0] = v.x; skyU[k][1] = v.y; skyU[k][2] = v.z; skyU[k][3] = w; };
    V3 camRight{rv.view.m[0], rv.view.m[4], rv.view.m[8]}, camUp{rv.view.m[1], rv.view.m[5], rv.view.m[9]};
    V3 camFwd{-rv.view.m[2], -rv.view.m[6], -rv.view.m[10]};
    sky4(0, camFwd, rv.time);
    sky4(1, camRight * (1.0f / rv.proj.m[0]), 0);
    sky4(2, camUp * (1.0f / rv.proj.m[5]), 0);
    sky4(3, env.sunDir, smoothstep(-0.03f, 0.02f, env.sunDir.y));
    sky4(4, env.sunColor, env.night);
    sky4(5, env.zenith, env.sunset);
    sky4(6, env.horizon, lerpf(1.9f, 1.45f, env.day));
    sky4(7, env.moonDir, 0);

    // OSD vertices must be written before the buffer is bound.
    const auto& osdVerts = osdLayer.data();
    int osdCount = std::min<int>(int(osdVerts.size()), Gpu::kOsdMaxVerts);
    if (osdCount > 0) {
        sg_write_buffer_desc wd = {};
        wd.src.data = {osdVerts.data(), size_t(osdCount) * sizeof(Osd::Vert)};
        wd.dst.buffer = g.osdBuf;
        sg_write_buffer_transient(&wd);
    }

    // Opaque first (by pipeline), then blended back to front.
    for (DrawItem& d : g.queue) {
        V3 c = d.xf.transformPoint((d.prim->bmin + d.prim->bmax) * 0.5f) - rv.camPos;
        d.depth = dot(c, c);
    }
    std::stable_sort(g.queue.begin(), g.queue.end(), [](const DrawItem& a, const DrawItem& b) {
        if (a.pass != b.pass) return passOrder(a.pass) < passOrder(b.pass);
        return a.pass == 2 ? a.depth > b.depth : false;
    });

    // --- Shadow cascades for the sun (or moon): spheres in front of the camera, fitted
    // to a light-space ortho box and snapped to whole texels so edges don't crawl.
    float shadowU[kShadowVec4][4] = {};
    const bool shadowsOn = (env.keyColor.x + env.keyColor.y + env.keyColor.z) > 0.01f;
    if (shadowsOn) {
        V3 L = normalize(env.keyDir);
        V3 camFwdFlat = normalize(V3{-rv.view.m[2], 0.0f, -rv.view.m[10]});
        const float radius[2] = {30.0f, 160.0f};
        for (int c = 0; c < 2; ++c) {
            V3 center = rv.camPos + camFwdFlat * (radius[c] * 0.6f);
            V3 up = std::fabs(L.y) > 0.95f ? V3{1, 0, 0} : V3{0, 1, 0};
            M4 view = lookAt(center + L * (radius[c] * 3.0f), center, up);
            float texel = 2.0f * radius[c] / kShadowSize;
            V3 o = view.transformPoint({0, 0, 0});
            view = translation({std::round(o.x / texel) * texel - o.x, std::round(o.y / texel) * texel - o.y, 0}) * view;
            M4 proj = ortho(-radius[c], radius[c], -radius[c], radius[c], 0.1f, radius[c] * 6.0f);
            M4 lightVP = proj * view;
            std::memcpy(shadowU[c * 4], lightVP.m, 64);
            shadowU[8][c] = texel;

            sg_pass sp = {};
            sp.action.depth.load_action = SG_LOADACTION_CLEAR;
            sp.action.depth.clear_value = 1.0f;
            sp.action.depth.store_action = SG_STOREACTION_STORE;
            sp.attachments.depth_stencil = g.shadowAtt[c];
            sg_begin_pass(&sp);
            sg_apply_pipeline(g.shadowPip);
            for (const DrawItem& d : g.queue) {
                if (d.pass == 2) continue;  // glass, ghosts and other blended draws cast no shadow
                M4 mvp = lightVP * d.xf;
                sg_apply_uniforms(0, {mvp.m, 64});
                float fs[4] = {d.mat->params[0][3], 0, 0, 0};
                sg_apply_uniforms(1, {fs, sizeof(fs)});
                sg_bindings b = {};
                b.vertex_buffers[0] = d.model->pos;
                b.vertex_buffers[1] = d.model->uv;
                b.index_buffer = d.model->idx;
                b.views[0] = d.mat->tex.id ? d.mat->tex : g.whiteTex;
                b.samplers[0] = g.materialSmp;
                sg_apply_bindings(&b);
                sg_draw(int(d.prim->firstIndex), int(d.prim->indexCount), 1);
            }
            sg_end_pass();
        }
        shadowU[8][2] = 1.0f / kShadowSize;
        shadowU[8][3] = 1.0f;
    }

    // --- Scene pass, multisampled, at the chosen fraction of window resolution.
    int tw = std::max(320, int(float(fbW) * rv.renderScale + 0.5f));
    int th = std::max(240, int(float(fbH) * rv.renderScale + 0.5f));
    g.ensureTarget(tw, th);
    sg_pass pass = {};
    pass.action.colors[0].load_action = SG_LOADACTION_CLEAR;
    pass.action.colors[0].clear_value = {0.010f, 0.012f, 0.018f, 1.0f};
    pass.action.colors[0].store_action = SG_STOREACTION_DONTCARE;
    pass.attachments.colors[0] = g.colorAtt;
    pass.attachments.resolves[0] = g.resolveAtt;
    pass.attachments.depth_stencil = g.depthAtt;
    sg_begin_pass(&pass);
    sg_apply_pipeline(g.skyPip);
    sg_apply_uniforms(0, {skyU, sizeof(skyU)});
    sg_draw(0, 3, 1);
    int currentPass = -1;
    for (const DrawItem& d : g.queue) {
        if (d.pass != currentPass) {
            currentPass = d.pass;
            sg_apply_pipeline(g.scenePip[currentPass]);
            sg_apply_uniforms(1, {frame, sizeof(frame)});
            sg_apply_uniforms(3, {shadowU, sizeof(shadowU)});
        }
        float vs[9][4] = {};
        M4 mvp = viewProj * d.xf;
        std::memcpy(vs[0], mvp.m, 64);
        std::memcpy(vs[4], d.xf.m, 64);
        vs[8][0] = d.emissive;
        vs[8][1] = d.alpha;
        sg_apply_uniforms(0, {vs, sizeof(vs)});
        sg_apply_uniforms(2, {d.mat->params, sizeof(d.mat->params)});
        sg_bindings b = {};
        b.vertex_buffers[0] = d.model->pos;
        b.vertex_buffers[1] = d.model->nrm;
        b.vertex_buffers[2] = d.model->uv;
        b.vertex_buffers[3] = d.model->tan;
        b.index_buffer = d.model->idx;
        b.views[0] = d.mat->tex.id ? d.mat->tex : g.whiteTex;
        b.views[1] = d.mat->detail.id ? d.mat->detail : g.whiteTex;
        b.views[2] = d.mat->normal.id ? d.mat->normal : g.whiteTex;
        b.views[3] = d.mat->maps.id ? d.mat->maps : g.whiteTex;
        b.views[4] = g.shadowTex[0];
        b.views[5] = g.shadowTex[1];
        b.samplers[0] = g.materialSmp;
        b.samplers[1] = g.shadowSmp;
        sg_apply_bindings(&b);
        sg_draw(int(d.prim->firstIndex), int(d.prim->indexCount), 1);
    }
    g.queue.clear();

    if (osdCount > 0) {
        sg_apply_pipeline(g.osdPip);
        float aspect = float(tw) / float(th), target = kOsdW / kOsdH;
        float ovs[4] = {aspect > target ? target / aspect : 1.0f, aspect > target ? 1.0f : aspect / target, 0, 0};
        sg_apply_uniforms(0, {ovs, sizeof(ovs)});
        sg_bindings b = {};
        b.vertex_buffers[0] = g.osdBuf;
        b.views[0] = g.fontTex;
        b.samplers[0] = g.nearestSmp;
        sg_apply_bindings(&b);
        sg_draw(0, osdCount, 1);
    }
    sg_end_pass();

    // --- Post pass to the window.
    sg_pass post = {};
    post.action.colors[0].load_action = SG_LOADACTION_CLEAR;
    post.action.colors[0].clear_value = {0, 0, 0, 1};
    post.swapchain.width = fbW;
    post.swapchain.height = fbH;
    post.swapchain.sample_count = 1;
    post.swapchain.color_format = SG_PIXELFORMAT_RGBA8;
    post.swapchain.depth_format = SG_PIXELFORMAT_DEPTH_STENCIL;
    post.swapchain.gl.framebuffer = 0;
    sg_begin_pass(&post);
    sg_apply_pipeline(g.postPip);
    float pvs[4] = {1.0f, 1.0f, 0, 0};
    float pfs[4] = {rv.time, rv.vhs, float(tw), float(th)};
    sg_apply_uniforms(0, {pvs, sizeof(pvs)});
    sg_apply_uniforms(1, {pfs, sizeof(pfs)});
    sg_bindings b = {};
    b.views[0] = g.colorTex;
    b.samplers[0] = g.linearSmp;
    sg_apply_bindings(&b);
    sg_draw(0, 6, 1);
    sg_end_pass();
    sg_commit();
}

// ---------------------------------------------------------------------------
// OSD primitives

void Osd::tri(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t c) {
    float u, v;
    solidUV(u, v);
    verts.push_back({x0, y0, u, v, c});
    verts.push_back({x1, y1, u, v, c});
    verts.push_back({x2, y2, u, v, c});
}

void Osd::rect(float x, float y, float w, float h, uint32_t c) {
    tri(x, y, x + w, y, x + w, y + h, c);
    tri(x, y, x + w, y + h, x, y + h, c);
}

void Osd::line(float x0, float y0, float x1, float y1, float width, uint32_t c) {
    float dx = x1 - x0, dy = y1 - y0, len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-4f) return;
    float nx = -dy / len * width * 0.5f, ny = dx / len * width * 0.5f;
    tri(x0 + nx, y0 + ny, x1 + nx, y1 + ny, x1 - nx, y1 - ny, c);
    tri(x0 + nx, y0 + ny, x1 - nx, y1 - ny, x0 - nx, y0 - ny, c);
}

void Osd::disc(float cx, float cy, float r, uint32_t c, int seg) {
    for (int k = 0; k < seg; ++k) {
        float a0 = 6.2831853f * k / seg, a1 = 6.2831853f * (k + 1) / seg;
        tri(cx, cy, cx + std::cos(a0) * r, cy + std::sin(a0) * r, cx + std::cos(a1) * r, cy + std::sin(a1) * r, c);
    }
}

void Osd::text(float x, float y, std::string_view s, uint32_t color, float scale, bool shadow) {
    auto emit = [&](float ox, float oy, uint32_t c) {
        float cx = ox;
        for (char ch : s) {
            int code = (unsigned char)ch;
            if (code >= 'a' && code <= 'z') code -= 32;
            if (ch != ' ') {
                float u0, v0, u1, v1;
                glyphUV(code, u0, v0, u1, v1);
                float w = 5 * scale, h = 7 * scale;
                verts.push_back({cx, oy, u0, v0, c});
                verts.push_back({cx + w, oy, u1, v0, c});
                verts.push_back({cx + w, oy + h, u1, v1, c});
                verts.push_back({cx, oy, u0, v0, c});
                verts.push_back({cx + w, oy + h, u1, v1, c});
                verts.push_back({cx, oy + h, u0, v1, c});
            }
            cx += 6 * scale;
        }
    };
    if (shadow) emit(x + scale * 0.5f, y + scale * 0.5f, rgba(0, 0, 0, 200));
    emit(x, y, color);
}

void Osd::gauge(float cx, float cy, float r, float value01, float redline01, int majorTicks, std::string_view label) {
    const uint32_t green = rgba(90, 255, 140, 255), red = rgba(255, 70, 50, 255);
    disc(cx, cy, r + 3, rgba(55, 55, 58, 255));
    disc(cx, cy, r, rgba(4, 6, 5, 240));
    auto polar = [&](float v, float rad, float& x, float& y) {
        float a = (225.0f - 270.0f * v) * 3.14159265f / 180.0f;
        x = cx + std::cos(a) * rad;
        y = cy - std::sin(a) * rad;
    };
    int minor = majorTicks * 5;
    for (int k = 0; k <= minor; ++k) {
        float v = float(k) / minor;
        bool major = k % 5 == 0;
        float x0, y0, x1, y1;
        polar(v, r * (major ? 0.72f : 0.82f), x0, y0);
        polar(v, r * 0.92f, x1, y1);
        line(x0, y0, x1, y1, major ? 1.6f : 0.8f, v >= redline01 ? red : green);
    }
    float nx, ny;
    polar(std::clamp(value01, 0.0f, 1.04f), r * 0.86f, nx, ny);
    line(cx, cy, nx, ny, 2.0f, rgba(255, 140, 50, 255));
    disc(cx, cy, 3.5f, rgba(40, 40, 40, 255), 12);
    float tw = float(label.size()) * 6.0f;
    text(cx - tw * 0.5f, cy + r * 0.38f, label, green, 1.0f, false);
}
