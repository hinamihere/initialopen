// sokol_gfx renderer: low-res night scene -> OSD overlay -> analog camcorder post pass.
#pragma once
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>
#include "assets.h"
#include "sky.h"
#include "vmath.h"

// Internal framebuffer: 4:3 standard definition, like a 90s Hi8 camcorder.
constexpr int kInternalW = 640;
constexpr int kInternalH = 480;

inline uint32_t rgba(int r, int g, int b, int a = 255) {
    return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
}

// 2D overlay in internal-framebuffer pixels (origin top-left). It is drawn
// before the tape effect, so it degrades along with the picture.
class Osd {
public:
    struct Vert {
        float x, y, u, v;
        uint32_t color;
    };
    void clear() { verts.clear(); }
    void text(float x, float y, std::string_view s, uint32_t color, float scale = 2.0f, bool shadow = true);
    void rect(float x, float y, float w, float h, uint32_t color);
    void line(float x0, float y0, float x1, float y1, float width, uint32_t color);
    void disc(float cx, float cy, float r, uint32_t color, int seg = 28);
    // Defi-style analog gauge: 270 degree sweep, green illumination.
    void gauge(float cx, float cy, float r, float value01, float redline01, int majorTicks, std::string_view label);
    const std::vector<Vert>& data() const { return verts; }

private:
    void tri(float x0, float y0, float x1, float y1, float x2, float y2, uint32_t c);
    std::vector<Vert> verts;
};

struct RenderView {
    M4 view, proj;
    V3 camPos;
    float time = 0;
    bool lightsOn = true;
    bool highBeam = false;
    float vhs = 1.0f;  // tape effect strength, 0 = clean
    V3 headlightPos[2];
    V3 beamDir{0, 0, 1};
    Environment env = environmentAt(23.0f);
};

class Renderer {
public:
    Renderer();
    ~Renderer();
    bool init();
    void shutdown();
    // Uploads a model to the GPU; returns its handle. CPU image data can be freed afterwards.
    int addModel(const ModelData& model);
    void setLights(const std::vector<TrackLight>* lights) { trackLights = lights; }
    // Queue one mesh group of a model for this frame.
    void submit(int model, int group, const M4& xf, float emissiveScale = 1.0f);
    Osd& osd() { return osdLayer; }
    void render(const RenderView& rv, int fbWidth, int fbHeight);

private:
    struct Gpu;
    std::unique_ptr<Gpu> gpu;
    const std::vector<TrackLight>* trackLights = nullptr;
    Osd osdLayer;
};
