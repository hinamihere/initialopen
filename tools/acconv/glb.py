"""Minimal glTF 2.0 binary (.glb) writer.

Images may be PNG or DDS. DDS images are stored with mimeType
"image/vnd-ms.dds" and referenced through the MSFT_texture_dds extension,
which keeps GPU-compressed textures compressed all the way to the engine.
"""
from __future__ import annotations

import json
import struct
from array import array

FLOAT, UINT32 = 5126, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


class Glb:
    def __init__(self, generator: str):
        self.bin = bytearray()
        self.doc = {
            "asset": {"version": "2.0", "generator": generator},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "buffers": [], "bufferViews": [], "accessors": [], "meshes": [],
            "nodes": [], "materials": [], "textures": [], "images": [], "samplers": [],
        }
        self._uses_dds = False

    def view(self, data: bytes, target: int | None = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        bv = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bin += data
        self.doc["bufferViews"].append(bv)
        return len(self.doc["bufferViews"]) - 1

    def _accessor(self, view: int, ctype: int, count: int, typ: str, **extra) -> int:
        acc = {"bufferView": view, "componentType": ctype, "count": count, "type": typ, **extra}
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def vec_accessor(self, values: array, comps: int, with_bounds: bool = False) -> int:
        count = len(values) // comps
        extra = {}
        if with_bounds and count:
            extra["min"] = [min(values[c::comps]) for c in range(comps)]
            extra["max"] = [max(values[c::comps]) for c in range(comps)]
        v = self.view(values.tobytes(), ARRAY_BUFFER)
        return self._accessor(v, FLOAT, count, {2: "VEC2", 3: "VEC3", 4: "VEC4"}[comps], **extra)

    def index_accessor(self, indices: array) -> int:
        v = self.view(indices.tobytes(), ELEMENT_ARRAY_BUFFER)
        return self._accessor(v, UINT32, len(indices), "SCALAR")

    def image(self, data: bytes, name: str) -> int:
        is_dds = data[:4] == b"DDS "
        mime = "image/vnd-ms.dds" if is_dds else ("image/png" if data[:4] == b"\x89PNG" else "image/jpeg")
        self.doc["images"].append({"name": name, "bufferView": self.view(data), "mimeType": mime})
        img = len(self.doc["images"]) - 1
        if not self.doc["samplers"]:
            self.doc["samplers"].append({"magFilter": 9729, "minFilter": 9987})
        tex = {"sampler": 0, "name": name}
        if is_dds:
            self._uses_dds = True
            tex["extensions"] = {"MSFT_texture_dds": {"source": img}}
        else:
            tex["source"] = img
        self.doc["textures"].append(tex)
        return len(self.doc["textures"]) - 1

    def material(self, mat: dict) -> int:
        self.doc["materials"].append(mat)
        return len(self.doc["materials"]) - 1

    def mesh(self, name: str, primitives: list) -> int:
        self.doc["meshes"].append({"name": name, "primitives": primitives})
        return len(self.doc["meshes"]) - 1

    def node(self, node: dict, root: bool = True) -> int:
        self.doc["nodes"].append(node)
        idx = len(self.doc["nodes"]) - 1
        if root:
            self.doc["scenes"][0]["nodes"].append(idx)
        return idx

    def write(self, path: str):
        if self._uses_dds:
            self.doc["extensionsUsed"] = ["MSFT_texture_dds"]
        for key in ("textures", "images", "samplers", "materials", "meshes"):
            if not self.doc[key]:
                del self.doc[key]
        while len(self.bin) % 4:
            self.bin.append(0)
        self.doc["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.doc, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, total))
            f.write(struct.pack("<I4s", len(js), b"JSON"))
            f.write(js)
            f.write(struct.pack("<I4s", len(self.bin), b"BIN\x00"))
            f.write(self.bin)
