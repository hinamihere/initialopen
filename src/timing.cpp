#include "timing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <type_traits>

namespace {

std::filesystem::path utf8Path(const std::string& s) {
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

constexpr float kStartGate = 1.0f;   // meters past the line where a standing start triggers
constexpr float kWrapWindow = 80.0f; // how close to the line a lap wrap must happen
constexpr int kCheckpoints = 24;     // anti-cut: all must be passed in order

}  // namespace

// --- TrackProgress ------------------------------------------------------------

void TrackProgress::init(const std::vector<V3>& line) {
    pts = line;
    hint = -1;
    if (pts.size() >= 2) {
        V3 d = pts.front() - pts.back();
        isClosed = ::length(d) < 20.0f;
        if (isClosed) pts.push_back(pts.front());
    }
    cum.assign(pts.size(), 0.0f);
    for (size_t k = 1; k < pts.size(); ++k) cum[k] = cum[k - 1] + ::length(pts[k] - pts[k - 1]);
    total = cum.empty() ? 0.0f : cum.back();
}

float TrackProgress::project(V3 p) {
    const int n = int(pts.size()) - 1;  // segment count
    if (n < 1) return 0.0f;
    float bestD = 1e30f, bestS = 0;
    int bestI = 0;
    auto test = [&](int i) {
        V3 a = pts[i], ab = pts[i + 1] - a;
        float len2 = std::max(dot(ab, ab), 1e-6f);
        float t = clampf(dot(p - a, ab) / len2, 0.0f, 1.0f);
        V3 d = p - (a + ab * t);
        float dist = d.x * d.x + d.z * d.z + d.y * d.y * 4.0f;  // height matters on stacked switchbacks
        if (dist < bestD) {
            bestD = dist;
            bestS = cum[i] + t * std::sqrt(len2);
            bestI = i;
        }
    };
    if (hint >= 0) {
        for (int k = -40; k <= 40; ++k) {
            int i = hint + k;
            if (isClosed) i = (i % n + n) % n;
            else if (i < 0 || i >= n) continue;
            test(i);
        }
    }
    if (hint < 0 || bestD > 30.0f * 30.0f)
        for (int i = 0; i < n; ++i) test(i);
    hint = bestI;
    return bestS;
}

// --- Run -----------------------------------------------------------------------

RunFrame Run::sample(float t) const {
    if (frames.empty()) return {};
    if (t <= frames.front().t) return frames.front();
    if (t >= frames.back().t) return frames.back();
    auto it = std::upper_bound(frames.begin(), frames.end(), t, [](float v, const RunFrame& f) { return v < f.t; });
    const RunFrame& b = *it;
    const RunFrame& a = *(it - 1);
    float u = (t - a.t) / std::max(b.t - a.t, 1e-6f);
    RunFrame f = a;
    f.t = t;
    f.progress = lerpf(a.progress, b.progress, u);
    f.pos = lerp(a.pos, b.pos, u);
    f.rot = nlerp(a.rot, b.rot, u);
    for (int w = 0; w < 4; ++w) {
        f.wheelPos[w] = lerp(a.wheelPos[w], b.wheelPos[w], u);
        f.wheelRot[w] = nlerp(a.wheelRot[w], b.wheelRot[w], u);
    }
    f.speed = lerpf(a.speed, b.speed, u);
    f.rpm = lerpf(a.rpm, b.rpm, u);
    f.steer = lerpf(a.steer, b.steer, u);
    return f;
}

float Run::timeAtProgress(float s) const {
    if (frames.empty()) return 0.0f;
    // Progress is monotonic over a clean run, so a binary search works.
    auto it = std::lower_bound(frames.begin(), frames.end(), s, [](const RunFrame& f, float v) { return f.progress < v; });
    if (it == frames.begin()) return frames.front().t;
    if (it == frames.end()) return frames.back().t;
    const RunFrame& b = *it;
    const RunFrame& a = *(it - 1);
    float u = (s - a.progress) / std::max(b.progress - a.progress, 1e-4f);
    return lerpf(a.t, b.t, clampf(u, 0.0f, 1.0f));
}

static_assert(std::is_trivially_copyable_v<RunFrame>);
static constexpr char kRunMagic[6] = {'I', 'O', 'R', 'U', 'N', 1};

bool Run::save(const std::string& path) const {
    std::error_code ec;
    std::filesystem::create_directories(utf8Path(path).parent_path(), ec);
    std::ofstream f(utf8Path(path), std::ios::binary);
    if (!f) return false;
    auto put = [&](const void* p, size_t n) { f.write(static_cast<const char*>(p), std::streamsize(n)); };
    auto putStr = [&](const std::string& s) {
        uint32_t n = uint32_t(s.size());
        put(&n, 4);
        put(s.data(), n);
    };
    uint32_t frameSize = sizeof(RunFrame), count = uint32_t(frames.size());
    put(kRunMagic, sizeof(kRunMagic));
    put(&frameSize, 4);
    putStr(track);
    putStr(car);
    put(&time, 4);
    put(sectors, sizeof(sectors));
    put(&count, 4);
    put(frames.data(), frames.size() * sizeof(RunFrame));
    return bool(f);
}

bool Run::load(const std::string& path) {
    std::ifstream f(utf8Path(path), std::ios::binary);
    if (!f) return false;
    auto get = [&](void* p, size_t n) { return bool(f.read(static_cast<char*>(p), std::streamsize(n))); };
    auto getStr = [&](std::string& s) {
        uint32_t n = 0;
        if (!get(&n, 4) || n > 4096) return false;
        s.resize(n);
        return get(s.data(), n);
    };
    char magic[sizeof(kRunMagic)];
    uint32_t frameSize = 0, count = 0;
    if (!get(magic, sizeof(magic)) || std::memcmp(magic, kRunMagic, sizeof(magic)) != 0) return false;
    if (!get(&frameSize, 4) || frameSize != sizeof(RunFrame)) return false;
    if (!getStr(track) || !getStr(car) || !get(&time, 4) || !get(sectors, sizeof(sectors)) || !get(&count, 4)) return false;
    if (count > 1000000) return false;
    frames.resize(count);
    return get(frames.data(), size_t(count) * sizeof(RunFrame));
}

// --- LapTimer ------------------------------------------------------------------------

void LapTimer::init(float trackLength, bool closedLoop) {
    total = trackLength;
    closed = closedLoop;
    checkpoints.clear();
    for (int k = 0; k < kCheckpoints; ++k) checkpoints.push_back(total * (float(k) + 0.5f) / kCheckpoints);
    stop();
}

void LapTimer::stop() {
    isRunning = false;
    valid = true;
    prevProgress = -1;
}

void LapTimer::begin(double at) {
    isRunning = true;
    valid = true;
    startTime = at;
    nextCheckpoint = 0;
    currentSector = 0;
    std::fill(std::begin(splits), std::end(splits), 0.0f);
}

LapTimer::Event LapTimer::update(float progress, double now) {
    if (total <= 0) return Event::None;
    if (prevProgress < 0) {
        prevProgress = progress;
        prevTime = now;
        return Event::None;
    }
    Event ev = Event::None;
    const float prev = prevProgress;
    bool wrappedForward = closed && prev > total - kWrapWindow && progress < kWrapWindow;
    bool wrappedBackward = closed && prev < kWrapWindow && progress > total - kWrapWindow;
    // Time at which the line was crossed, interpolated between frames.
    auto crossTime = [&](float before, float after) {
        float frac = before / std::max(before + after, 1e-4f);
        return prevTime + (now - prevTime) * double(clampf(frac, 0.0f, 1.0f));
    };

    if (isRunning) {
        if (wrappedBackward) valid = false;  // reversed over the line
        // Anti-cut: checkpoints must be passed in order without big jumps.
        while (nextCheckpoint < kCheckpoints && !wrappedForward && progress >= checkpoints[nextCheckpoint]) {
            if (progress > checkpoints[nextCheckpoint] + 60.0f) valid = false;
            ++nextCheckpoint;
        }
        if (currentSector < kSectors - 1 && progress >= total * float(currentSector + 1) / kSectors &&
            prev < total * float(currentSector + 1) / kSectors && !wrappedForward) {
            splits[currentSector] = float(now - startTime);
            ++currentSector;
            ev = Event::Sector;
        }
        bool finished = closed ? wrappedForward : (progress >= total - 2.0f && prev < total - 2.0f);
        if (finished) {
            double at = closed ? crossTime(total - prev, progress) : now;
            lastTime = float(at - startTime);
            lastValid = valid && nextCheckpoint >= kCheckpoints;
            std::copy(std::begin(splits), std::end(splits), lastSplits);
            lastSplits[kSectors - 1] = lastTime;
            ev = Event::Finished;
            if (closed) begin(at);  // flying lap: the next one starts right away
            else isRunning = false;
        }
    } else {
        bool standing = prev < kStartGate && progress >= kStartGate && progress < kWrapWindow;
        if (wrappedForward) {
            begin(crossTime(total - prev, progress));
            ev = Event::Started;
        } else if (standing) {
            begin(now);
            ev = Event::Started;
        }
    }
    prevProgress = progress;
    prevTime = now;
    return ev;
}

std::string formatLapTime(float seconds) {
    if (seconds < 0) seconds = 0;
    int ms = int(std::lround(seconds * 1000.0));
    char buf[32];
    std::snprintf(buf, sizeof buf, "%d:%02d.%03d", ms / 60000, ms / 1000 % 60, ms % 1000);
    return buf;
}
