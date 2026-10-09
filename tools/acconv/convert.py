"""acconv: convert Assetto Corsa cars and tracks into engine assets.

    python tools/acconv/convert.py car   <ac car folder>   <out dir>
    python tools/acconv/convert.py track <ac track folder> <layout> <out dir>

Outputs a .glb (visuals, plus collision meshes for tracks) and a .json with
everything else the game needs, extracted as faithfully as possible from the
mod's own data files. Interpretation (e.g. mapping AC diff lock to Jolt) is
left to the engine. Pure Python, no dependencies.

Assetto Corsa model space is +X left, +Y up, +Z forward, the same as the engine,
so no axis conversion happens here. KN5 matrices are row-major with row
vectors, which has the same memory layout as glTF's column-major matrices.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kn5  # noqa: E402
from glb import Glb  # noqa: E402

MAX_TEXTURE = 1024  # we render at 640x480; larger mips are dropped
IDENT = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

# Defaults from AC's system surfaces.ini; a track's own surfaces.ini overrides them.
DEFAULT_SURFACES = {
    "ROAD": {"friction": 0.98, "valid": True},
    "KERB": {"friction": 0.92, "valid": True},
    "GRASS": {"friction": 0.60, "valid": False},
    "SAND": {"friction": 0.60, "valid": False},
    "WALL": {"friction": 0.30, "valid": False},
}

# --------------------------------------------------------------------------
# INI / LUT


def read_ini(path: str) -> dict:
    sections: dict = {}
    if not os.path.exists(path):
        return sections
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = re.split(r";|//", raw, maxsplit=1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and "]" in line:
                cur = line[1:line.index("]")].strip().upper()
                sections.setdefault(cur, {})
            elif "=" in line and cur is not None:
                k, v = line.split("=", 1)
                sections[cur][k.strip().upper()] = v.strip()
    return sections


def num(s, default=0.0) -> float:
    try:
        return float(str(s).split()[0].rstrip(","))
    except (ValueError, IndexError):
        return default


def vec(s) -> list:
    return [float(x) for x in str(s).replace(" ", "").split(",") if x]


def read_lut(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.split(";")[0].strip()
            if "|" in line:
                a, b = line.split("|", 1)
                out.append([num(a), num(b)])
    return out


# --------------------------------------------------------------------------
# Matrices (row-major, row vectors: p' = p * M)


def mat_mul(a, b):
    return tuple(sum(a[r * 4 + j] * b[j * 4 + c] for j in range(4)) for r in range(4) for c in range(4))


def mat_translate(m, t):
    m = list(m)
    m[12] += t[0]
    m[13] += t[1]
    m[14] += t[2]
    return tuple(m)


def mat_inverse_affine(m):
    a = [[m[r * 4 + c] for c in range(3)] for r in range(3)]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    inv = [[0.0] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            r1, r2 = [i for i in range(3) if i != c]
            c1, c2 = [i for i in range(3) if i != r]
            cof = a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]
            inv[r][c] = (-1) ** (r + c) * cof / det
    t = [m[12], m[13], m[14]]
    ti = [-sum(t[j] * inv[j][c] for j in range(3)) for c in range(3)]
    return (inv[0][0], inv[0][1], inv[0][2], 0.0, inv[1][0], inv[1][1], inv[1][2], 0.0,
            inv[2][0], inv[2][1], inv[2][2], 0.0, ti[0], ti[1], ti[2], 1.0)


def walk_world(node, parent=IDENT, active=True):
    """Yield (node, world matrix, effective active flag)."""
    world = mat_mul(node.matrix, parent) if node.kind == 1 else parent
    active = active and node.active
    yield node, world, active
    for c in node.children:
        yield from walk_world(c, world, active)


# --------------------------------------------------------------------------
# Geometry


def bake(mesh, M):
    """Transform a kn5 mesh by M. Returns (positions, normals, uvs, tangents, indices16)."""
    a = array("f")
    a.frombytes(mesh.vertices)
    n = mesh.vertex_count
    px, py, pz = a[0::11], a[1::11], a[2::11]
    nx, ny, nz = a[3::11], a[4::11], a[5::11]
    qx, qy, qz = a[8::11], a[9::11], a[10::11]
    pos, nrm, uv = array("f", bytes(12 * n)), array("f", bytes(12 * n)), array("f", bytes(8 * n))
    tan = array("f", bytes(12 * n))
    if M == IDENT:
        pos[0::3], pos[1::3], pos[2::3] = px, py, pz
        nrm[0::3], nrm[1::3], nrm[2::3] = nx, ny, nz
        tan[0::3], tan[1::3], tan[2::3] = qx, qy, qz
    else:
        m = M
        pos[0::3] = array("f", [x * m[0] + y * m[4] + z * m[8] + m[12] for x, y, z in zip(px, py, pz)])
        pos[1::3] = array("f", [x * m[1] + y * m[5] + z * m[9] + m[13] for x, y, z in zip(px, py, pz)])
        pos[2::3] = array("f", [x * m[2] + y * m[6] + z * m[10] + m[14] for x, y, z in zip(px, py, pz)])
        tx = [x * m[0] + y * m[4] + z * m[8] for x, y, z in zip(nx, ny, nz)]
        ty = [x * m[1] + y * m[5] + z * m[9] for x, y, z in zip(nx, ny, nz)]
        tz = [x * m[2] + y * m[6] + z * m[10] for x, y, z in zip(nx, ny, nz)]
        inv = [1.0 / max(math.sqrt(x * x + y * y + z * z), 1e-12) for x, y, z in zip(tx, ty, tz)]
        nrm[0::3] = array("f", [x * s for x, s in zip(tx, inv)])
        nrm[1::3] = array("f", [y * s for y, s in zip(ty, inv)])
        nrm[2::3] = array("f", [z * s for z, s in zip(tz, inv)])
        tan[0::3] = array("f", [x * m[0] + y * m[4] + z * m[8] for x, y, z in zip(qx, qy, qz)])
        tan[1::3] = array("f", [x * m[1] + y * m[5] + z * m[9] for x, y, z in zip(qx, qy, qz)])
        tan[2::3] = array("f", [x * m[2] + y * m[6] + z * m[10] for x, y, z in zip(qx, qy, qz)])
    uv[0::2], uv[1::2] = a[6::11], a[7::11]
    idx = array("H")
    idx.frombytes(mesh.indices)
    return pos, nrm, uv, tan, idx


class Batch:
    """Geometry merged for one material."""

    def __init__(self):
        self.pos, self.nrm, self.uv, self.tan, self.idx = array("f"), array("f"), array("f"), array("f"), array("I")

    def add(self, pos, nrm, uv, idx, tan=None):
        base = len(self.pos) // 3
        self.pos += pos
        if nrm is not None:
            self.nrm += nrm
            self.uv += uv
            self.tan += tan
        self.idx += array("I", [i + base for i in idx])

    @property
    def tris(self):
        return len(self.idx) // 3


def shrink_dds(d: bytes) -> bytes:
    """Drop top mip levels larger than MAX_TEXTURE (no re-encoding)."""
    if d[:4] != b"DDS ":
        return d
    h, w = struct.unpack_from("<2I", d, 12)
    mips = max(1, struct.unpack_from("<I", d, 28)[0])
    fourcc = d[84:88]
    bitcount = struct.unpack_from("<I", d, 88)[0]
    if fourcc == b"DXT1":
        size = lambda w, h: max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 8  # noqa: E731
    elif fourcc in (b"DXT3", b"DXT5"):
        size = lambda w, h: max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16  # noqa: E731
    elif fourcc == b"\0\0\0\0" and bitcount in (24, 32):
        size = lambda w, h: w * h * bitcount // 8  # noqa: E731
    else:
        return d
    off = 128
    while max(w, h) > MAX_TEXTURE and mips > 1:
        off += size(w, h)
        w, h, mips = max(1, w // 2), max(1, h // 2), mips - 1
    if off == 128:
        return d
    hdr = bytearray(d[:128])
    struct.pack_into("<3I", hdr, 12, h, w, size(w, h))
    struct.pack_into("<I", hdr, 28, mips)
    return bytes(hdr) + d[off:]


class MaterialExporter:
    def __init__(self, glb: Glb, emissive_overrides: dict | None = None, texture_overrides: dict | None = None):
        self.glb = glb
        self.texture_overrides = texture_overrides or {}  # lowercase name -> bytes (car skins)
        self.tex_cache: dict = {}
        self.mat_cache: dict = {}
        self.emissive_overrides = emissive_overrides or {}
        self.texture_bytes = 0

    def texture(self, model: kn5.Kn5, model_id, name: str):
        key = (model_id, name)
        if key not in self.tex_cache:
            tex = next((t for t in model.textures if t.name == name), None)
            override = self.texture_overrides.get(name.lower())
            if override is not None:
                tex = kn5.Texture(name, override)
            if tex is None or not tex.data:
                self.tex_cache[key] = None
            else:
                data = shrink_dds(tex.data)
                self.texture_bytes += len(data)
                self.tex_cache[key] = self.glb.image(data, name)
        return self.tex_cache[key]

    def get(self, model: kn5.Kn5, model_id, mat_index: int, emissive=None) -> int:
        key = (model_id, mat_index, emissive)
        if key in self.mat_cache:
            return self.mat_cache[key]
        m = model.materials[mat_index]
        props = {k: v for k, v in m.props.items()}
        ks_emissive = list(props.get("ksEmissive", (0, (0, 0), (0, 0, 0), (0, 0, 0, 0)))[2])
        if m.name in self.emissive_overrides:
            ks_emissive = self.emissive_overrides[m.name]
        if emissive is not None:
            ks_emissive = list(emissive)
        alpha_test = m.alpha_tested or "AT" in m.shader
        blend = m.blend_mode == 1
        mat = {
            "name": m.name,
            "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "alphaMode": "BLEND" if blend else ("MASK" if alpha_test else "OPAQUE"),
            "doubleSided": alpha_test or blend,
            "emissiveFactor": [min(1.0, max(0.0, c)) for c in ks_emissive],
            "extras": {
                "ac_shader": m.shader,
                "ksDiffuse": props.get("ksDiffuse", (0.4,))[0],
                "ksAmbient": props.get("ksAmbient", (0.4,))[0],
                "ksSpecular": props.get("ksSpecular", (0.0,))[0],
                "ksSpecularEXP": props.get("ksSpecularEXP", (20.0,))[0],
                "fresnelC": props.get("fresnelC", (0.0,))[0],
                "fresnelEXP": props.get("fresnelEXP", (5.0,))[0],
                "fresnelMaxLevel": props.get("fresnelMaxLevel", (0.0,))[0],
                "isAdditive": props.get("isAdditive", (0.0,))[0],
                "ksEmissive": ks_emissive,
            },
        }
        if alpha_test:
            mat["alphaCutoff"] = 0.5
        # ksPerPixelMultiMap: txDiffuse alpha blends between the diffuse and txDetail
        # (on car bodies txDetail is the skin's paint color).
        if m.shader.startswith("ksPerPixelMultiMap") and "txDetail" in m.samplers:
            dt = self.texture(model, model_id, m.samplers["txDetail"])
            if dt is not None:
                mat["extras"]["detail_texture"] = dt
                mat["extras"]["detail_uv"] = props.get("detailUVMultiplier", (1.0,))[0]
        # Normal map (tangent space, Direct3D green channel) and the multimap "maps"
        # texture: R = specular, G = gloss, B = reflection mask.
        nm = m.samplers.get("txNormal")
        if nm and not nm.lower().startswith("flat"):
            t = self.texture(model, model_id, nm)
            if t is not None:
                mat["normalTexture"] = {"index": t}
        if m.shader.startswith("ksPerPixelMultiMap") and "txMaps" in m.samplers:
            t = self.texture(model, model_id, m.samplers["txMaps"])
            if t is not None:
                mat["extras"]["maps_texture"] = t
        tex_name = m.samplers.get("txDiffuse")
        if tex_name:
            t = self.texture(model, model_id, tex_name)
            if t is not None:
                mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": t}
        idx = self.glb.material(mat)
        self.mat_cache[key] = idx
        return idx


def emit_batches(glb: Glb, name: str, batches: dict, matx: MaterialExporter, extra_node: dict | None = None,
                 root=True):
    prims = []
    for (model, model_id, mat_index, emissive), b in batches.items():
        if not b.idx:
            continue
        n = len(b.tan) // 3
        tan4 = array("f", bytes(16 * n))
        tan4[0::4], tan4[1::4], tan4[2::4] = b.tan[0::3], b.tan[1::3], b.tan[2::3]
        tan4[3::4] = array("f", [1.0]) * n
        prims.append({
            "attributes": {
                "POSITION": glb.vec_accessor(b.pos, 3, with_bounds=True),
                "NORMAL": glb.vec_accessor(b.nrm, 3),
                "TANGENT": glb.vec_accessor(tan4, 4),
                "TEXCOORD_0": glb.vec_accessor(b.uv, 2),
            },
            "indices": glb.index_accessor(b.idx),
            "material": matx.get(model, model_id, mat_index, emissive),
        })
    if not prims:
        return None
    node = {"name": name, "mesh": glb.mesh(name, prims)}
    if extra_node:
        node.update(extra_node)
    return glb.node(node, root)


def mb(path):
    return os.path.getsize(path) / (1024 * 1024)


# --------------------------------------------------------------------------
# Track


def cluster_points(points, radius):
    cells: dict = {}
    clusters = []  # [sumx, sumy, sumz, count]
    for p in points:
        key = (int(p[0] // radius), int(p[1] // radius), int(p[2] // radius))
        found = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for ci in cells.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        c = clusters[ci]
                        cx, cy, cz = c[0] / c[3], c[1] / c[3], c[2] / c[3]
                        if (cx - p[0]) ** 2 + (cy - p[1]) ** 2 + (cz - p[2]) ** 2 < radius * radius:
                            found = ci
                            break
                    if found is not None:
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            clusters.append([0.0, 0.0, 0.0, 0])
            found = len(clusters) - 1
            cells.setdefault(key, []).append(found)
        c = clusters[found]
        c[0] += p[0]
        c[1] += p[1]
        c[2] += p[2]
        c[3] += 1
    return [[c[0] / c[3], c[1] / c[3], c[2] / c[3]] for c in clusters]


def csp_color(values):
    """CSP colors are either 0..255 with an optional multiplier, or raw HDR floats."""
    rgb = values[:3]
    mult = values[3] if len(values) > 3 else 1.0
    if max(rgb) > 1.0:
        rgb = [c / 255.0 for c in rgb]
    return [c * mult for c in rgb]


def read_ai_line(path):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        d = f.read()
    _version, count, _lap, _samples = struct.unpack_from("<4i", d, 0)
    return [struct.unpack_from("<3f", d, 16 + 20 * i) for i in range(count)]


def convert_track(track_dir: str, layout: str, out_dir: str):
    t0 = time.time()
    track_id = os.path.basename(os.path.normpath(track_dir))
    layout_dir = os.path.join(track_dir, layout) if layout else track_dir
    models_ini = read_ini(os.path.join(track_dir, f"models_{layout}.ini" if layout else "models.ini"))
    files = []
    for sec in sorted((s for s in models_ini if s.startswith("MODEL_")), key=lambda s: int(s.split("_")[1])):
        e = models_ini[sec]
        if any(abs(v) > 1e-6 for v in vec(e.get("ROTATION", "0,0,0"))):
            print(f"  warning: {e['FILE']} has a ROTATION, which is not supported yet; ignoring it")
        files.append((e["FILE"], vec(e.get("POSITION", "0,0,0")) or [0, 0, 0]))
    if not files:
        files = [(f"{track_id}.kn5", [0, 0, 0])]

    surfaces = {k: dict(v) for k, v in DEFAULT_SURFACES.items()}
    for path in (os.path.join(track_dir, "data", "surfaces.ini"), os.path.join(layout_dir, "data", "surfaces.ini")):
        for sec in read_ini(path).values():
            if "KEY" in sec:
                surfaces[sec["KEY"].upper()] = {"friction": num(sec.get("FRICTION", 0.9)),
                                                "valid": num(sec.get("IS_VALID_TRACK", 0)) > 0}

    # CSP config: night emissives for lamp glass, and which materials emit light.
    ext = read_ini(os.path.join(track_dir, "extension", "ext_config.ini"))
    emissive_overrides, light_series = {}, []
    for name, sec in ext.items():
        mats = [m.strip() for m in sec.get("MATERIALS", "").split(",") if m.strip()]
        if name.startswith("MATERIAL_ADJUSTMENT") and sec.get("KEY_0", "").lower() == "ksemissive":
            for m in mats:
                emissive_overrides[m] = csp_color(vec(sec.get("VALUE_0", "0,0,0")))
        if name.startswith("LIGHT_SERIES") and mats:
            light_series.append({
                "materials": set(mats),
                "color": csp_color(vec(sec.get("COLOR", "255,200,150"))),
                "range": num(sec.get("RANGE", 30), 30),
                "spot": num(sec.get("SPOT", 0)),
                "points": [],
            })

    glb = Glb("initialopen acconv")
    matx = MaterialExporter(glb, emissive_overrides)
    batches: dict = {}
    collision: dict = {}
    markers = []
    stats = {"meshes": 0, "hidden": 0}
    for fi, (fname, offset) in enumerate(files):
        path = os.path.join(track_dir, fname)
        if not os.path.exists(path):
            print(f"  warning: missing {fname}")
            continue
        model = kn5.read(path)
        base = mat_translate(IDENT, offset)
        for node, W, active in walk_world(model.root, base):
            upper = node.name.upper()
            if upper.startswith("AC_"):
                if node.kind == 1:
                    markers.append((node.name, W))
                continue
            if not node.mesh:
                continue
            mesh = node.mesh
            m = re.match(r"^\d+([A-Z]+)", node.name)
            surface = m.group(1) if m and m.group(1) in surfaces else None
            visible = active and mesh.visible and mesh.renderable
            if not visible and not surface:
                stats["hidden"] += 1
                continue
            pos, nrm, uv, tan, idx = bake(mesh, W)
            if surface:
                collision.setdefault(surface, Batch()).add(pos, None, None, idx)
            if not visible:
                continue
            stats["meshes"] += 1
            batches.setdefault((model, fi, mesh.material, None), Batch()).add(pos, nrm, uv, idx, tan)
            mat_name = model.materials[mesh.material].name
            for s in light_series:
                if mat_name in s["materials"]:
                    s["points"] += list(zip(pos[0::3], pos[1::3], pos[2::3]))
        print(f"  {fname}: done ({time.time() - t0:.1f}s)")

    visual = batches
    emit_batches(glb, "TRACK", visual, matx)
    for key, b in sorted(collision.items()):
        prim = {"attributes": {"POSITION": glb.vec_accessor(b.pos, 3, with_bounds=True)}, "indices": glb.index_accessor(b.idx)}
        glb.node({"name": f"COLLISION_{key}", "mesh": glb.mesh(f"COLLISION_{key}", [prim]),
                  "extras": {"collision": True, "surface": key}})

    # Driving direction from the AI line; spawn markers take their facing from it.
    ai = read_ai_line(os.path.join(layout_dir, "ai", "fast_lane.ai"))

    def ai_dir_near(p):
        if len(ai) < 2:
            return None
        i = min(range(len(ai)), key=lambda k: (ai[k][0] - p[0]) ** 2 + (ai[k][2] - p[2]) ** 2)
        a, b = ai[max(i - 1, 0)], ai[min(i + 1, len(ai) - 1)]
        d = [b[0] - a[0], 0.0, b[2] - a[2]]
        n = math.hypot(d[0], d[2]) or 1.0
        return [d[0] / n, 0.0, d[2] / n]

    # Find which local axis of the markers is "forward" by comparing the hotlap
    # start marker with the racing line, then apply it to every marker.
    axis_sign = 1.0
    for name, W in markers:
        if name.upper().startswith(("AC_HOTLAP_START", "AC_START")):
            d = ai_dir_near(W[12:15])
            if d:
                axis_sign = 1.0 if W[8] * d[0] + W[10] * d[2] >= 0 else -1.0
            break
    spawns = []
    for name, W in markers:
        f = [W[8] * axis_sign, 0.0, W[10] * axis_sign]
        n = math.hypot(f[0], f[2]) or 1.0
        spawns.append({"name": name, "pos": [round(v, 4) for v in W[12:15]], "fwd": [f[0] / n, 0.0, f[2] / n]})
    spawns.sort(key=lambda s: s["name"])

    lights = []
    for s in light_series:
        for c in cluster_points(s["points"], 2.0):
            lights.append({"pos": [round(v, 3) for v in c], "dir": [0, -1, 0], "color": s["color"],
                           "range": s["range"], "spot_deg": s["spot"]})

    ui_name = track_id
    ui_path = os.path.join(track_dir, "ui", layout, "ui_track.json") if layout else os.path.join(track_dir, "ui", "ui_track.json")
    try:
        with open(ui_path, encoding="utf-8", errors="replace") as f:
            ui_name = json.loads(f.read(), strict=False).get("name", track_id)
    except (OSError, ValueError):
        pass

    length = sum(math.dist(ai[i], ai[i + 1]) for i in range(len(ai) - 1)) if ai else 0.0
    meta = {
        "format": 1,
        "source": {"type": "assetto_corsa_track", "id": track_id, "layout": layout},
        "name": ui_name,
        "length": round(length, 1),
        "surfaces": surfaces,
        "spawns": spawns,
        "lights": lights,
        "ai_line": [round(v, 3) for p in ai[::2] for v in p],
    }
    os.makedirs(out_dir, exist_ok=True)
    glb_path = os.path.join(out_dir, "track.glb")
    glb.write(glb_path)
    with open(os.path.join(out_dir, "track.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    vis_tris = sum(b.tris for b in visual.values())
    print(f"track '{ui_name}' [{layout}] -> {out_dir}")
    print(f"  visual: {vis_tris:,} tris in {len(visual)} batches from {stats['meshes']} meshes ({stats['hidden']} hidden skipped)")
    print(f"  textures: {len(matx.tex_cache)} ({matx.texture_bytes / 2**20:.1f} MB after mip trim)")
    print("  collision: " + ", ".join(f"{k} {b.tris:,} tris" for k, b in sorted(collision.items())))
    print(f"  spawns: {len(spawns)}, lights: {len(lights)}, ai line: {len(ai)} pts / {length:.0f} m")
    print(f"  track.glb {mb(glb_path):.1f} MB, {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------
# Car

WHEELS = ("WHEEL_LF", "WHEEL_RF", "WHEEL_LR", "WHEEL_RR")
SUSPS = ("SUSP_LF", "SUSP_RF", "SUSP_LR", "SUSP_RR")


def pick_skin(car_dir: str, wanted: str | None):
    skins_dir = os.path.join(car_dir, "skins")
    skins = sorted(d for d in os.listdir(skins_dir) if os.path.isdir(os.path.join(skins_dir, d))) if os.path.isdir(skins_dir) else []
    if not skins:
        return None, {}
    if wanted:
        match = [d for d in skins if d.lower() == wanted.lower()]
        if not match:
            sys.exit(f"skin '{wanted}' not found; available: {', '.join(skins)}")
        skin = match[0]
    else:
        skin = skins[0]
    folder = os.path.join(skins_dir, skin)
    files = {}
    for f in os.listdir(folder):
        if f.lower().endswith((".dds", ".png", ".jpg", ".bmp", ".tga")) and f.lower() != "preview.jpg":
            with open(os.path.join(folder, f), "rb") as fh:
                files[f.lower()] = fh.read()
    return skin, files


def convert_car(car_dir: str, out_dir: str, skin_name: str | None = None):
    t0 = time.time()
    car_id = os.path.basename(os.path.normpath(car_dir))
    data = os.path.join(car_dir, "data")
    if not os.path.isdir(data):
        if os.path.exists(os.path.join(car_dir, "data.acd")):
            sys.exit("this car only ships an encrypted data.acd; unpacking it is not supported")
        sys.exit(f"no data folder in {car_dir}")
    ini = {n: read_ini(os.path.join(data, f"{n}.ini")) for n in
           ("car", "suspensions", "tyres", "engine", "drivetrain", "brakes", "lights", "colliders", "aero")}
    car, susp, tyres = ini["car"], ini["suspensions"], ini["tyres"]
    gfx_offset = vec(car.get("BASIC", {}).get("GRAPHICS_OFFSET", "0,0,0")) or [0, 0, 0]

    kn5_path = os.path.join(car_dir, f"{car_id}.kn5")
    if not os.path.exists(kn5_path):
        cands = [f for f in os.listdir(car_dir) if f.lower().endswith(".kn5") and "collider" not in f.lower()]
        kn5_path = os.path.join(car_dir, max(cands, key=lambda f: os.path.getsize(os.path.join(car_dir, f))))
    model = kn5.read(kn5_path)

    head_names = {s["NAME"] for k, s in ini["lights"].items() if k.startswith("LIGHT_") and "NAME" in s}
    brake_names = {s["NAME"] for k, s in ini["lights"].items() if k.startswith("BRAKE_") and "NAME" in s}
    light_color = {s["NAME"]: vec(s.get("COLOR", "1,1,1")) for k, s in ini["lights"].items() if "NAME" in s}

    def emissive_for(name):
        # AC light colors are HDR multipliers: ~2000 for headlights, ~1-5 for
        # gauge backlights. Compress that range into 0..1 emissive strength.
        c = light_color.get(name) or [1, 1, 1]
        peak = max(c) or 1.0
        strength = peak / (peak + 60.0)
        return tuple(round(v / peak * strength, 4) for v in c[:3])

    # Assign every mesh to a group; record group frames (body space).
    groups: dict = {}
    frames = {}
    offset_m = mat_translate(IDENT, gfx_offset)

    def visit(node, parent_W, active, group, light):
        W = mat_mul(node.matrix, parent_W) if node.kind == 1 else parent_W
        active = active and node.active
        name = node.name
        if name in WHEELS or name in SUSPS or name == "STEER_HR":
            group = name
            frames[name] = mat_mul(W, offset_m)
        if name in head_names:
            light = ("LIGHTS_ON", name)
        elif name in brake_names:
            light = ("LIGHTS_BRAKE", name)
        if node.mesh and active and node.mesh.visible and node.mesh.renderable:
            g, emissive = group, None
            if light and group in ("BODY",):
                g, emissive = light[0], emissive_for(light[1])
            groups.setdefault(g, []).append((node, mat_mul(W, offset_m), emissive))
        for c in node.children:
            visit(c, W, active, group, light)

    visit(model.root, IDENT, True, "BODY", None)

    skin, skin_files = pick_skin(car_dir, skin_name)
    glb = Glb("initialopen acconv")
    matx = MaterialExporter(glb, texture_overrides=skin_files)
    tri_count = 0
    for gname, items in groups.items():
        frame = frames.get(gname)
        if gname.startswith(("WHEEL_", "SUSP_")):
            frame = mat_translate(IDENT, frame[12:15])  # position only: physics drives orientation
        to_local = mat_inverse_affine(frame) if frame else IDENT
        batches: dict = {}
        for node, W, emissive in items:
            pos, nrm, uv, tan, idx = bake(node.mesh, mat_mul(W, to_local) if frame else W)
            key = (model, 0, node.mesh.material, emissive)
            batches.setdefault(key, Batch()).add(pos, nrm, uv, idx, tan)
        extra = {"matrix": list(frame)} if frame else None
        emit_batches(glb, gname, batches, matx, extra)
        tri_count += sum(b.tris for b in batches.values())

    # Physical data, in body space (origin at the center of gravity).
    basic = susp.get("BASIC", {})
    wheelbase, cg = num(basic.get("WHEELBASE", 2.5)), num(basic.get("CG_LOCATION", 0.5))
    wheels = []
    for name in ("LF", "RF", "LR", "RR"):
        front = name[1] == "F"
        s = susp.get("FRONT" if front else "REAR", {})
        t = tyres.get("FRONT" if front else "REAR", {})
        side = 1.0 if name[0] == "L" else -1.0
        visual = frames.get("WHEEL_" + name)
        wheels.append({
            "name": name,
            "pos": [side * num(s.get("TRACK", 1.5)) / 2, num(s.get("BASEY", -0.2)),
                    wheelbase * (1 - cg) if front else -wheelbase * cg],
            "visual_pos": [round(v, 4) for v in visual[12:15]] if visual else None,
            "radius": num(t.get("RADIUS", 0.3)),
            "width": num(t.get("WIDTH", 0.2)),
            "rim_radius": num(t.get("RIM_RADIUS", 0.2)),
            "steer": front,
            "spring_rate": num(s.get("SPRING_RATE", 60000)),
            "damp_bump": num(s.get("DAMP_BUMP", 3000)),
            "damp_rebound": num(s.get("DAMP_REBOUND", 4000)),
            "packer_range": num(s.get("PACKER_RANGE", 0.1)),
            "bumpstop_up": num(s.get("BUMPSTOP_UP", 0.06)),
            "bumpstop_dn": num(s.get("BUMPSTOP_DN", 0.06)),
            "static_camber_deg": num(s.get("STATIC_CAMBER", 0)),
            "tyre": {k.lower(): num(t[k]) for k in ("DX0", "DY0", "DX_REF", "DY_REF", "FRICTION_LIMIT_ANGLE", "FLEX") if k in t},
            "tyre_name": t.get("NAME", ""),
        })

    eng = ini["engine"]
    turbos = [{"max_boost": num(s.get("MAX_BOOST")), "wastegate": num(s.get("WASTEGATE", s.get("MAX_BOOST"))),
               "reference_rpm": num(s.get("REFERENCE_RPM", 4000)), "gamma": num(s.get("GAMMA", 1))}
              for k, s in sorted(eng.items()) if k.startswith("TURBO_")]
    dt = ini["drivetrain"]
    gears = dt.get("GEARS", {})
    gear_count = int(num(gears.get("COUNT", 5)))
    brakes = ini["brakes"].get("DATA", {})
    graphics = car.get("GRAPHICS", {})
    eyes = vec(graphics.get("DRIVEREYES", "0,1,0"))
    colliders = [{"center": vec(s.get("CENTRE", "0,0,0")), "size": vec(s.get("SIZE", "1,1,1"))}
                 for k, s in sorted(ini["colliders"].items()) if k.startswith("COLLIDER_")]
    # collider.kn5 is the body shape AC uses against walls; keep its bounds (body space).
    hull = None
    hull_path = os.path.join(car_dir, "collider.kn5")
    if os.path.exists(hull_path):
        pts = []
        for node, W, _ in walk_world(kn5.read(hull_path, load_textures=False).root):
            if node.mesh:
                p, _, _, _, _ = bake(node.mesh, mat_mul(W, offset_m))
                pts += list(zip(p[0::3], p[1::3], p[2::3]))
        if pts:
            hull = {"min": [round(min(q[i] for q in pts), 4) for i in range(3)],
                    "max": [round(max(q[i] for q in pts), 4) for i in range(3)]}
    info = car.get("INFO", {})
    meta = {
        "format": 1,
        "source": {"type": "assetto_corsa_car", "id": car_id},
        "name": info.get("SCREEN_NAME", car_id),
        "skin": skin,
        "mass": num(car.get("BASIC", {}).get("TOTALMASS", 1200)),
        "inertia_box": vec(car.get("BASIC", {}).get("INERTIA", "1.8,1.2,4.2")),
        "graphics_offset": gfx_offset,
        "driver_eyes": [eyes[0] + gfx_offset[0], eyes[1] + gfx_offset[1], eyes[2] + gfx_offset[2]],
        "onboard_pitch_deg": num(graphics.get("ON_BOARD_PITCH_ANGLE", 0)),
        "colliders": colliders,
        "collider_bounds": hull,
        "wheels": wheels,
        "arb": {"front": num(susp.get("ARB", {}).get("FRONT", 0)), "rear": num(susp.get("ARB", {}).get("REAR", 0))},
        "steer": {"lock_deg": num(car.get("CONTROLS", {}).get("STEER_LOCK", 400)),
                  "ratio": abs(num(car.get("CONTROLS", {}).get("STEER_RATIO", 14)))},
        "engine": {
            "idle_rpm": num(eng.get("ENGINE_DATA", {}).get("MINIMUM", 900)),
            "limiter_rpm": num(eng.get("ENGINE_DATA", {}).get("LIMITER", 7000)),
            "inertia": num(eng.get("ENGINE_DATA", {}).get("INERTIA", 0.1)),
            "power_lut": read_lut(os.path.join(data, eng.get("HEADER", {}).get("POWER_CURVE", "power.lut"))),
            "coast_torque": num(eng.get("COAST_REF", {}).get("TORQUE", 30)),
            "turbos": turbos,
        },
        "drivetrain": {
            "type": dt.get("TRACTION", {}).get("TYPE", "RWD").upper(),
            "gears": [num(gears.get(f"GEAR_{i}")) for i in range(1, gear_count + 1)],
            "reverse": num(gears.get("GEAR_R", -3.0)),
            "final": num(gears.get("FINAL", 4.0)),
            "diff": {k.lower(): num(v) for k, v in dt.get("DIFFERENTIAL", {}).items()},
            "awd": {k.lower(): num(v) for k, v in dt.get("AWD", {}).items()},
            "shift_up_ms": num(dt.get("GEARBOX", {}).get("CHANGE_UP_TIME", 200)),
            "shift_down_ms": num(dt.get("GEARBOX", {}).get("CHANGE_DN_TIME", 250)),
        },
        "brakes": {"max_torque": num(brakes.get("MAX_TORQUE", 2000)), "front_share": num(brakes.get("FRONT_SHARE", 0.6)),
                   "handbrake_torque": num(brakes.get("HANDBRAKE_TORQUE", 800))},
    }
    os.makedirs(out_dir, exist_ok=True)
    # Sound: the mod's FMOD Studio bank plus GUIDs.txt (event ids; there is no strings bank).
    sfx_src, sfx_dst = os.path.join(car_dir, "sfx"), os.path.join(out_dir, "sfx")
    sfx_files = [f for f in os.listdir(sfx_src) if f.lower().endswith(".bank") or f == "GUIDs.txt"] if os.path.isdir(sfx_src) else []
    if sfx_files:
        os.makedirs(sfx_dst, exist_ok=True)
        for f in sfx_files:
            shutil.copyfile(os.path.join(sfx_src, f), os.path.join(sfx_dst, f))
    meta["sound_bank"] = next((f"sfx/{f}" for f in sfx_files if f.lower().endswith(".bank")), None)
    glb_path = os.path.join(out_dir, "car.glb")
    glb.write(glb_path)
    with open(os.path.join(out_dir, "car.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"car '{meta['name']}' -> {out_dir}")
    print(f"  skin: {skin} ({len(skin_files)} textures)")
    print(f"  {tri_count:,} tris, groups: {', '.join(sorted(groups))}")
    print(f"  textures: {len(matx.tex_cache)} ({matx.texture_bytes / 2**20:.1f} MB after mip trim)")
    print(f"  {meta['mass']:.0f} kg, {meta['drivetrain']['type']}, {gear_count} gears, "
          f"{len(meta['engine']['power_lut'])}-point torque curve, {len(turbos)} turbo(s)")
    print(f"  sound: {meta['sound_bank'] or 'none'}")
    print(f"  car.glb {mb(glb_path):.1f} MB, {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("car")
    c.add_argument("car_dir")
    c.add_argument("out_dir")
    c.add_argument("--skin", help="skin folder name (default: first alphabetically)")
    t = sub.add_parser("track")
    t.add_argument("track_dir")
    t.add_argument("layout", help="layout folder name, or '' for single-layout tracks")
    t.add_argument("out_dir")
    a = ap.parse_args()
    if a.cmd == "car":
        convert_car(a.car_dir, a.out_dir, a.skin)
    else:
        convert_track(a.track_dir, a.layout, a.out_dir)


if __name__ == "__main__":
    main()
