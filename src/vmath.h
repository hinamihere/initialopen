// Minimal float vector/matrix math for rendering and track generation.
// Matrices are column-major (OpenGL convention).
#pragma once
#include <cmath>

struct V3 {
    float x = 0, y = 0, z = 0;
    V3() = default;
    constexpr V3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    V3 operator+(V3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    V3 operator-(V3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    V3 operator*(float s) const { return {x * s, y * s, z * s}; }
    V3 operator-() const { return {-x, -y, -z}; }
    V3& operator+=(V3 b) { x += b.x; y += b.y; z += b.z; return *this; }
};
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 normalize(V3 a) { float l = length(a); return l > 1e-8f ? a * (1.0f / l) : V3{0, 1, 0}; }
inline V3 lerp(V3 a, V3 b, float t) { return a + (b - a) * t; }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float smoothstep(float e0, float e1, float x) {
    float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

struct M4 {
    float m[16];
    static M4 identity() {
        M4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
    M4 operator*(const M4& b) const {
        M4 r{};
        for (int c = 0; c < 4; ++c)
            for (int rr = 0; rr < 4; ++rr) {
                float s = 0;
                for (int k = 0; k < 4; ++k) s += m[k * 4 + rr] * b.m[c * 4 + k];
                r.m[c * 4 + rr] = s;
            }
        return r;
    }
    V3 transformPoint(V3 p) const {
        return {m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
                m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
                m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]};
    }
    V3 transformDir(V3 d) const {
        return {m[0] * d.x + m[4] * d.y + m[8] * d.z,
                m[1] * d.x + m[5] * d.y + m[9] * d.z,
                m[2] * d.x + m[6] * d.y + m[10] * d.z};
    }
};

inline M4 perspective(float fovyRad, float aspect, float zn, float zf) {
    M4 r{};
    float f = 1.0f / std::tan(fovyRad * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zf + zn) / (zn - zf);
    r.m[11] = -1.0f;
    r.m[14] = 2.0f * zf * zn / (zn - zf);
    return r;
}

inline M4 lookAt(V3 eye, V3 target, V3 up) {
    V3 f = normalize(target - eye);
    V3 s = normalize(cross(f, up));
    V3 u = cross(s, f);
    M4 r = M4::identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

struct Quat {
    float x = 0, y = 0, z = 0, w = 1;
};

// Rotation part of a rigid transform as a quaternion.
inline Quat quatFromMat(const M4& m) {
    float r00 = m.m[0], r11 = m.m[5], r22 = m.m[10];
    float r10 = m.m[1], r20 = m.m[2], r01 = m.m[4], r21 = m.m[6], r02 = m.m[8], r12 = m.m[9];
    Quat q;
    float tr = r00 + r11 + r22;
    if (tr > 0) {
        float s = 0.5f / std::sqrt(tr + 1.0f);
        q = {(r21 - r12) * s, (r02 - r20) * s, (r10 - r01) * s, 0.25f / s};
    } else if (r00 > r11 && r00 > r22) {
        float s = 2.0f * std::sqrt(1.0f + r00 - r11 - r22);
        q = {0.25f * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s};
    } else if (r11 > r22) {
        float s = 2.0f * std::sqrt(1.0f + r11 - r00 - r22);
        q = {(r01 + r10) / s, 0.25f * s, (r12 + r21) / s, (r02 - r20) / s};
    } else {
        float s = 2.0f * std::sqrt(1.0f + r22 - r00 - r11);
        q = {(r02 + r20) / s, (r12 + r21) / s, 0.25f * s, (r10 - r01) / s};
    }
    return q;
}

inline M4 matFromQuat(Quat q, V3 p) {
    M4 m = M4::identity();
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z, xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    m.m[0] = 1 - 2 * (yy + zz); m.m[4] = 2 * (xy - wz);     m.m[8] = 2 * (xz + wy);
    m.m[1] = 2 * (xy + wz);     m.m[5] = 1 - 2 * (xx + zz); m.m[9] = 2 * (yz - wx);
    m.m[2] = 2 * (xz - wy);     m.m[6] = 2 * (yz + wx);     m.m[10] = 1 - 2 * (xx + yy);
    m.m[12] = p.x; m.m[13] = p.y; m.m[14] = p.z;
    return m;
}

// Normalized lerp along the shorter arc.
inline Quat nlerp(Quat a, Quat b, float t) {
    if (a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w < 0) b = {-b.x, -b.y, -b.z, -b.w};
    Quat q{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t};
    float l = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return {q.x / l, q.y / l, q.z / l, q.w / l};
}

// Inverse of a rotation + translation matrix.
inline M4 rigidInverse(const M4& m) {
    M4 r = M4::identity();
    for (int c = 0; c < 3; ++c)
        for (int rr = 0; rr < 3; ++rr) r.m[c * 4 + rr] = m.m[rr * 4 + c];
    V3 t{m.m[12], m.m[13], m.m[14]};
    r.m[12] = -(r.m[0] * t.x + r.m[4] * t.y + r.m[8] * t.z);
    r.m[13] = -(r.m[1] * t.x + r.m[5] * t.y + r.m[9] * t.z);
    r.m[14] = -(r.m[2] * t.x + r.m[6] * t.y + r.m[10] * t.z);
    return r;
}

inline M4 ortho(float l, float r, float b, float t, float n, float f) {
    M4 m = M4::identity();
    m.m[0] = 2.0f / (r - l);
    m.m[5] = 2.0f / (t - b);
    m.m[10] = -2.0f / (f - n);
    m.m[12] = -(r + l) / (r - l);
    m.m[13] = -(t + b) / (t - b);
    m.m[14] = -(f + n) / (f - n);
    return m;
}

inline M4 translation(V3 t) {
    M4 r = M4::identity();
    r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
    return r;
}

inline M4 scaling(V3 s) {
    M4 r = M4::identity();
    r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z;
    return r;
}
