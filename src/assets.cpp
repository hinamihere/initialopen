#include "assets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include "stb_image.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

// Textures larger than this are downsampled when we decode them anyway.
constexpr int kMaxTextureSize = 1024;

// --- RGBA8 helpers ----------------------------------------------------------

std::vector<uint8_t> halve(const std::vector<uint8_t>& src, int w, int h, int& nw, int& nh) {
    nw = std::max(1, w / 2);
    nh = std::max(1, h / 2);
    std::vector<uint8_t> dst(size_t(nw) * nh * 4);
    for (int y = 0; y < nh; ++y)
        for (int x = 0; x < nw; ++x)
            for (int c = 0; c < 4; ++c) {
                int x0 = std::min(x * 2, w - 1), x1 = std::min(x * 2 + 1, w - 1);
                int y0 = std::min(y * 2, h - 1), y1 = std::min(y * 2 + 1, h - 1);
                int s = src[(size_t(y0) * w + x0) * 4 + c] + src[(size_t(y0) * w + x1) * 4 + c] +
                        src[(size_t(y1) * w + x0) * 4 + c] + src[(size_t(y1) * w + x1) * 4 + c];
                dst[(size_t(y) * nw + x) * 4 + c] = uint8_t((s + 2) / 4);
            }
    return dst;
}

void finishRgba(ImageData& img, std::vector<uint8_t> px, int w, int h) {
    while (std::max(w, h) > kMaxTextureSize) {
        int nw, nh;
        px = halve(px, w, h, nw, nh);
        w = nw;
        h = nh;
    }
    img.format = ImageData::RGBA8;
    img.width = w;
    img.height = h;
    img.mips.clear();
    img.mips.push_back(px);
    while (w > 1 || h > 1) {
        int nw, nh;
        img.mips.push_back(halve(img.mips.back(), w, h, nw, nh));
        w = nw;
        h = nh;
    }
}

// --- DDS --------------------------------------------------------------------

uint32_t rd32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

int maskShift(uint32_t m) {
    if (!m) return 0;
    int s = 0;
    while (!(m & 1)) { m >>= 1; ++s; }
    return s;
}

bool decodeDds(const uint8_t* d, size_t size, ImageData& img) {
    if (size < 128 || std::memcmp(d, "DDS ", 4) != 0) return false;
    int h = int(rd32(d + 12)), w = int(rd32(d + 16));
    int mips = std::max(1, int(rd32(d + 28)));
    uint32_t pfFlags = rd32(d + 80);
    const uint8_t* fourcc = d + 84;
    uint32_t bits = rd32(d + 88);
    size_t off = 128;
    int blockBytes = 0;
    ImageData::Format fmt = ImageData::RGBA8;
    if (std::memcmp(fourcc, "DXT1", 4) == 0) { fmt = ImageData::BC1; blockBytes = 8; }
    else if (std::memcmp(fourcc, "DXT3", 4) == 0) { fmt = ImageData::BC2; blockBytes = 16; }
    else if (std::memcmp(fourcc, "DXT5", 4) == 0) { fmt = ImageData::BC3; blockBytes = 16; }
    else if (std::memcmp(fourcc, "DX10", 4) == 0) {
        if (size < 148) return false;
        uint32_t dxgi = rd32(d + 128);
        off = 148;
        if (dxgi == 71 || dxgi == 72) { fmt = ImageData::BC1; blockBytes = 8; }
        else if (dxgi == 74 || dxgi == 75) { fmt = ImageData::BC2; blockBytes = 16; }
        else if (dxgi == 77 || dxgi == 78) { fmt = ImageData::BC3; blockBytes = 16; }
        else return false;
    }

    if (blockBytes) {
        img.format = fmt;
        img.width = w;
        img.height = h;
        int lw = w, lh = h;
        for (int m = 0; m < mips; ++m) {
            size_t ls = size_t(std::max(1, (lw + 3) / 4)) * std::max(1, (lh + 3) / 4) * blockBytes;
            if (off + ls > size) break;
            img.mips.emplace_back(d + off, d + off + ls);
            off += ls;
            lw = std::max(1, lw / 2);
            lh = std::max(1, lh / 2);
        }
        return !img.mips.empty();
    }

    // Uncompressed: decode the top level by channel masks, then build our own mips.
    if (!(pfFlags & 0x40) || (bits != 32 && bits != 24)) return false;
    uint32_t mr = rd32(d + 92), mg = rd32(d + 96), mb = rd32(d + 100), ma = (pfFlags & 0x1) ? rd32(d + 104) : 0;
    int bpp = int(bits / 8);
    if (off + size_t(w) * h * bpp > size) return false;
    std::vector<uint8_t> px(size_t(w) * h * 4);
    int sr = maskShift(mr), sg = maskShift(mg), sb = maskShift(mb), sa = maskShift(ma);
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        uint32_t v = 0;
        std::memcpy(&v, d + off + i * bpp, size_t(bpp));
        px[i * 4 + 0] = uint8_t((v & mr) >> sr);
        px[i * 4 + 1] = uint8_t((v & mg) >> sg);
        px[i * 4 + 2] = uint8_t((v & mb) >> sb);
        px[i * 4 + 3] = ma ? uint8_t((v & ma) >> sa) : 255;
    }
    finishRgba(img, std::move(px), w, h);
    return true;
}

bool decodeImage(const uint8_t* d, size_t size, ImageData& img) {
    if (size >= 4 && std::memcmp(d, "DDS ", 4) == 0) return decodeDds(d, size, img);
    int w, h, n;
    stbi_uc* px = stbi_load_from_memory(d, int(size), &w, &h, &n, 4);
    if (!px) return false;
    finishRgba(img, std::vector<uint8_t>(px, px + size_t(w) * h * 4), w, h);
    stbi_image_free(px);
    return true;
}

// Paths are UTF-8 everywhere; narrow-string file APIs on Windows would use the ANSI code page.
std::filesystem::path utf8Path(const std::string& s) {
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

bool readFile(const std::string& path, std::vector<char>& out) {
    std::ifstream f(utf8Path(path), std::ios::binary | std::ios::ate);
    if (!f) return false;
    out.resize(size_t(f.tellg()));
    f.seekg(0);
    return bool(f.read(out.data(), std::streamsize(out.size())));
}

// --- glTF helpers -------------------------------------------------------------

const uint8_t* accessorData(const cgltf_accessor* a) {
    return static_cast<const uint8_t*>(a->buffer_view->buffer->data) + a->buffer_view->offset + a->offset;
}

void readFloats(const cgltf_accessor* a, int comps, std::vector<float>& out) {
    size_t base = out.size();
    out.resize(base + a->count * comps);
    if (a->component_type == cgltf_component_type_r_32f && (a->stride == size_t(comps) * 4) && a->buffer_view) {
        std::memcpy(out.data() + base, accessorData(a), a->count * comps * 4);
    } else {
        cgltf_accessor_unpack_floats(a, out.data() + base, a->count * comps);
    }
}

int textureImage(const cgltf_data* data, const cgltf_texture* tex) {
    if (!tex) return -1;
    if (tex->image) return int(tex->image - data->images);
    for (cgltf_size e = 0; e < tex->extensions_count; ++e) {
        const cgltf_extension& ext = tex->extensions[e];
        if (ext.name && std::strcmp(ext.name, "MSFT_texture_dds") == 0 && ext.data) {
            json j = json::parse(ext.data, nullptr, false);
            if (j.is_object() && j.contains("source")) return j["source"].get<int>();
        }
    }
    return -1;
}

V3 v3(const json& j) { return {j.at(0).get<float>(), j.at(1).get<float>(), j.at(2).get<float>()}; }

}  // namespace

int ModelData::findGroup(const std::string& name) const {
    for (size_t k = 0; k < groups.size(); ++k)
        if (groups[k].name == name) return int(k);
    return -1;
}

bool loadGlb(const std::string& path, ModelData& out, std::vector<CollisionData>* collision, std::string& err) {
    std::vector<char> file;
    if (!readFile(path, file)) {
        err = "cannot open " + path;
        return false;
    }
    cgltf_options opt = {};
    cgltf_data* data = nullptr;
    if (cgltf_parse(&opt, file.data(), file.size(), &data) != cgltf_result_success) {
        err = "cannot parse " + path;
        return false;
    }
    // Self-contained .glb: buffers come from the embedded BIN chunk, no file access.
    if (cgltf_load_buffers(&opt, data, nullptr) != cgltf_result_success) {
        err = "cannot load buffers of " + path;
        cgltf_free(data);
        return false;
    }

    out.images.resize(data->images_count);
    for (cgltf_size i = 0; i < data->images_count; ++i) {
        const cgltf_image& im = data->images[i];
        if (!im.buffer_view) continue;
        const uint8_t* bytes = static_cast<const uint8_t*>(im.buffer_view->buffer->data) + im.buffer_view->offset;
        if (!decodeImage(bytes, im.buffer_view->size, out.images[i]))
            std::fprintf(stderr, "warning: could not decode image '%s'\n", im.name ? im.name : "?");
    }

    for (cgltf_size i = 0; i < data->materials_count; ++i) {
        const cgltf_material& m = data->materials[i];
        MaterialData md;
        md.name = m.name ? m.name : "";
        if (m.has_pbr_metallic_roughness) {
            std::memcpy(md.baseColor, m.pbr_metallic_roughness.base_color_factor, sizeof(md.baseColor));
            md.image = textureImage(data, m.pbr_metallic_roughness.base_color_texture.texture);
        }
        std::memcpy(md.emissive, m.emissive_factor, sizeof(md.emissive));
        if (m.extras.data) {
            json ex = json::parse(m.extras.data, nullptr, false);
            if (ex.is_object() && ex.contains("detail_texture")) {
                int t = ex["detail_texture"].get<int>();
                if (t >= 0 && t < int(data->textures_count)) md.detailImage = textureImage(data, &data->textures[t]);
                md.detailUv = ex.value("detail_uv", 1.0f);
            }
        }
        md.alpha = m.alpha_mode == cgltf_alpha_mode_blend  ? MaterialData::Blend
                   : m.alpha_mode == cgltf_alpha_mode_mask ? MaterialData::Mask
                                                           : MaterialData::Opaque;
        md.cutoff = m.alpha_cutoff;
        md.doubleSided = m.double_sided;
        out.materials.push_back(md);
    }

    for (cgltf_size n = 0; n < data->nodes_count; ++n) {
        const cgltf_node& node = data->nodes[n];
        if (!node.mesh) continue;
        std::string name = node.name ? node.name : "";
        if (name.rfind("COLLISION_", 0) == 0) {
            if (!collision) continue;
            for (cgltf_size p = 0; p < node.mesh->primitives_count; ++p) {
                const cgltf_primitive& prim = node.mesh->primitives[p];
                CollisionData cd;
                cd.surface = name.substr(10);
                std::vector<float> pos;
                for (cgltf_size a = 0; a < prim.attributes_count; ++a)
                    if (prim.attributes[a].type == cgltf_attribute_type_position) readFloats(prim.attributes[a].data, 3, pos);
                cd.v.resize(pos.size() / 3);
                for (size_t k = 0; k < cd.v.size(); ++k) cd.v[k] = {pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2]};
                if (prim.indices) {
                    cd.i.resize(prim.indices->count);
                    for (cgltf_size k = 0; k < prim.indices->count; ++k) cd.i[k] = uint32_t(cgltf_accessor_read_index(prim.indices, k));
                }
                collision->push_back(std::move(cd));
            }
            continue;
        }
        MeshGroup g;
        g.name = name;
        cgltf_node_transform_local(&node, g.matrix.m);
        for (cgltf_size p = 0; p < node.mesh->primitives_count; ++p) {
            const cgltf_primitive& prim = node.mesh->primitives[p];
            const cgltf_accessor *pa = nullptr, *na = nullptr, *ta = nullptr;
            for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
                const cgltf_attribute& at = prim.attributes[a];
                if (at.type == cgltf_attribute_type_position) pa = at.data;
                else if (at.type == cgltf_attribute_type_normal) na = at.data;
                else if (at.type == cgltf_attribute_type_texcoord && at.index == 0) ta = at.data;
            }
            if (!pa || !prim.indices) continue;
            uint32_t base = uint32_t(out.pos.size() / 3);
            readFloats(pa, 3, out.pos);
            if (na) readFloats(na, 3, out.nrm); else out.nrm.resize(out.pos.size(), 0.0f);
            if (ta) readFloats(ta, 2, out.uv); else out.uv.resize(out.pos.size() / 3 * 2, 0.0f);
            Primitive pr;
            pr.firstIndex = uint32_t(out.idx.size());
            pr.indexCount = uint32_t(prim.indices->count);
            pr.material = prim.material ? int(prim.material - data->materials) : -1;
            pr.bmin = {pa->min[0], pa->min[1], pa->min[2]};
            pr.bmax = {pa->max[0], pa->max[1], pa->max[2]};
            out.idx.resize(out.idx.size() + prim.indices->count);
            uint32_t* dst = out.idx.data() + pr.firstIndex;
            if (prim.indices->component_type == cgltf_component_type_r_32u && prim.indices->stride == 4) {
                std::memcpy(dst, accessorData(prim.indices), prim.indices->count * 4);
                for (cgltf_size k = 0; k < prim.indices->count; ++k) dst[k] += base;
            } else {
                for (cgltf_size k = 0; k < prim.indices->count; ++k) dst[k] = base + uint32_t(cgltf_accessor_read_index(prim.indices, k));
            }
            g.prims.push_back(pr);
        }
        out.groups.push_back(std::move(g));
    }
    cgltf_free(data);
    return true;
}

const Spawn* TrackInfo::findSpawn(const std::string& prefix) const {
    for (const Spawn& s : spawns)
        if (s.name.rfind(prefix, 0) == 0) return &s;
    return nullptr;
}

static bool readJson(const std::string& path, json& j, std::string& err) {
    std::vector<char> text;
    if (!readFile(path, text)) {
        err = "cannot open " + path;
        return false;
    }
    j = json::parse(text.begin(), text.end(), nullptr, false);
    if (j.is_discarded()) {
        err = "invalid JSON in " + path;
        return false;
    }
    return true;
}

bool loadTrackInfo(const std::string& path, TrackInfo& t, std::string& err) {
    json j;
    if (!readJson(path, j, err)) return false;
    try {
        t.name = j.value("name", "");
        t.length = j.value("length", 0.0f);
        for (auto& [k, v] : j.at("surfaces").items()) t.friction[k] = v.value("friction", 0.9f);
        for (auto& s : j.at("spawns")) t.spawns.push_back({s.at("name").get<std::string>(), v3(s.at("pos")), v3(s.at("fwd"))});
        for (auto& l : j.at("lights")) {
            V3 c = v3(l.at("color"));
            t.lights.push_back({v3(l.at("pos")), c, l.value("range", 30.0f)});
        }
        const auto& ai = j.at("ai_line");
        for (size_t k = 0; k + 2 < ai.size(); k += 3) t.aiLine.push_back({ai[k].get<float>(), ai[k + 1].get<float>(), ai[k + 2].get<float>()});
    } catch (const std::exception& e) {
        err = path + ": " + e.what();
        return false;
    }
    return true;
}

bool loadCarSpec(const std::string& path, CarSpec& c, std::string& err) {
    json j;
    if (!readJson(path, j, err)) return false;
    try {
        c.name = j.value("name", "");
        c.mass = j.at("mass").get<float>();
        c.inertiaBox = v3(j.at("inertia_box"));
        c.driverEyes = v3(j.at("driver_eyes"));
        c.onboardPitchDeg = j.value("onboard_pitch_deg", 0.0f);
        if (j.contains("collider_bounds") && j["collider_bounds"].is_object()) {
            c.hullMin = v3(j["collider_bounds"].at("min"));
            c.hullMax = v3(j["collider_bounds"].at("max"));
        }
        const auto& wheels = j.at("wheels");
        for (int k = 0; k < 4; ++k) {
            const auto& w = wheels.at(k);
            WheelSpec& ws = c.wheels[k];
            ws.name = w.at("name").get<std::string>();
            ws.pos = v3(w.at("pos"));
            ws.radius = w.at("radius").get<float>();
            ws.width = w.at("width").get<float>();
            ws.steer = w.value("steer", false);
            ws.springRate = w.value("spring_rate", 60000.0f);
            ws.dampBump = w.value("damp_bump", 3000.0f);
            ws.dampRebound = w.value("damp_rebound", 4000.0f);
            ws.packerRange = w.value("packer_range", 0.1f);
            const auto& t = w.at("tyre");
            ws.dx0 = t.value("dx0", 1.2f);
            ws.dy0 = t.value("dy0", 1.2f);
            ws.frictionLimitAngle = t.value("friction_limit_angle", 8.0f);
        }
        c.arbFront = j.at("arb").value("front", 0.0f);
        c.arbRear = j.at("arb").value("rear", 0.0f);
        c.steerLockDeg = j.at("steer").value("lock_deg", 400.0f);
        c.steerRatio = j.at("steer").value("ratio", 14.0f);

        const auto& e = j.at("engine");
        c.idleRpm = e.value("idle_rpm", 900.0f);
        c.limiterRpm = e.value("limiter_rpm", 7000.0f);
        c.engineInertia = e.value("inertia", 0.1f);
        // Assetto Corsa's turbo model: torque * (1 + boost), boost rising toward the
        // reference rpm and capped by the wastegate. Summed over all turbos.
        for (const auto& p : e.at("power_lut")) {
            float rpm = p.at(0).get<float>(), nm = p.at(1).get<float>();
            float boost = 0;
            for (const auto& t : e.at("turbos")) {
                float ref = std::max(1.0f, t.value("reference_rpm", 4000.0f));
                float b = t.value("max_boost", 0.0f) * std::pow(std::min(1.0f, rpm / ref), t.value("gamma", 1.0f));
                boost += std::min(b, t.value("wastegate", b));
            }
            c.torqueCurve.push_back({rpm, nm * (1.0f + boost)});
        }
        const auto& d = j.at("drivetrain");
        c.traction = d.value("type", "RWD");
        c.gears = d.at("gears").get<std::vector<float>>();
        c.reverseGear = d.value("reverse", -3.0f);
        c.finalDrive = d.value("final", 4.0f);
        c.shiftUpMs = d.value("shift_up_ms", 200.0f);
        float power = d.at("diff").value("power", 0.0f);
        if (c.traction == "AWD" && d.contains("awd")) power = std::max(power, d["awd"].value("rear_diff_power", 0.0f));
        c.diffPower = power;
        const auto& b = j.at("brakes");
        c.brakeTorque = b.value("max_torque", 2000.0f);
        c.brakeFrontShare = b.value("front_share", 0.6f);
        c.handbrakeTorque = b.value("handbrake_torque", 800.0f);
    } catch (const std::exception& ex) {
        err = path + ": " + ex.what();
        return false;
    }
    return true;
}
