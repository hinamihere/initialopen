// Time attack: distance along the racing line, lap/sector timing with cut
// detection, and recorded runs (ghosts now, replays later).
#pragma once
#include <string>
#include <vector>
#include "vmath.h"

// Projects positions onto the track's racing line (meters from its start).
class TrackProgress {
public:
    void init(const std::vector<V3>& line);
    float project(V3 p);
    float length() const { return total; }
    bool closed() const { return isClosed; }

private:
    std::vector<V3> pts;
    std::vector<float> cum;
    float total = 0;
    bool isClosed = false;
    int hint = -1;
};

// One recorded sample of a car.
struct RunFrame {
    float t = 0;         // seconds since the run started
    float progress = 0;  // meters along the racing line
    V3 pos;
    Quat rot;
    V3 wheelPos[4];      // wheel transforms relative to the body
    Quat wheelRot[4];
    float speed = 0, rpm = 0, steer = 0, throttle = 0, brake = 0;
    int gear = 0;
};

struct Run {
    std::string track, car;
    float time = 0;
    float sectors[3] = {};  // cumulative split times
    std::vector<RunFrame> frames;

    bool empty() const { return frames.size() < 2; }
    RunFrame sample(float t) const;          // interpolated
    float timeAtProgress(float s) const;     // when this run reached `s` meters
    bool save(const std::string& path) const;
    bool load(const std::string& path);
};

class LapTimer {
public:
    static constexpr int kSectors = 3;
    enum class Event { None, Started, Sector, Finished };

    void init(float trackLength, bool closedLoop);
    // Call once per frame with the car's progress and the simulation time.
    Event update(float progress, double now);
    void invalidate() { valid = false; }
    void stop();  // back to waiting for the start line (e.g. after teleporting)

    bool running() const { return isRunning; }
    bool isValid() const { return valid; }
    float lapTime(double now) const { return isRunning ? float(now - startTime) : 0.0f; }
    int sector() const { return currentSector; }
    float split(int k) const { return splits[k]; }   // cumulative, valid once passed
    float finishedTime() const { return lastTime; }   // time of the lap just finished
    bool finishedValid() const { return lastValid; }
    const float* finishedSplits() const { return lastSplits; }

private:
    float total = 0;
    bool closed = true;
    bool isRunning = false, valid = true;
    double startTime = 0, prevTime = 0;
    float prevProgress = -1;
    int nextCheckpoint = 0, currentSector = 0;
    std::vector<float> checkpoints;
    float splits[kSectors] = {};
    float lastTime = 0, lastSplits[kSectors] = {};
    bool lastValid = false;
    void begin(double at);
};

std::string formatLapTime(float seconds);  // m:ss.mmm
