"""Reader for Assetto Corsa .kn5 model files.

Layout (little endian):
  magic "sc6969", int32 version (+ int32 extra when version > 5)
  textures:  int32 count, then {int32 active, string name, int32 size, bytes}
  materials: int32 count, then {string name, string shader, uint8 blend,
             uint8 alphaTested, int32 depthMode, properties, samplers}
  nodes:     one root node, recursively {int32 class, string name,
             int32 childCount, uint8 active, class payload, children}
Strings are int32 length + UTF-8 bytes. Node transforms are row-major
(row-vector convention, translation in elements 12..14).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


class Reader:
    def __init__(self, data: bytes):
        self.d = memoryview(data)
        self.p = 0

    def take(self, n: int) -> memoryview:
        if self.p + n > len(self.d):
            raise ValueError(f"read past end at {self.p} (+{n})")
        v = self.d[self.p:self.p + n]
        self.p += n
        return v

    def i32(self) -> int:
        return struct.unpack_from("<i", self.take(4))[0]

    def u8(self) -> int:
        return self.take(1)[0]

    def f32(self, n: int = 1):
        v = struct.unpack_from(f"<{n}f", self.take(4 * n))
        return v[0] if n == 1 else v

    def string(self) -> str:
        n = self.i32()
        return bytes(self.take(n)).decode("utf-8", errors="replace")


@dataclass
class Texture:
    name: str
    data: bytes


@dataclass
class Material:
    name: str
    shader: str
    blend_mode: int
    alpha_tested: bool
    depth_mode: int
    props: dict = field(default_factory=dict)      # name -> (A, B2, C3, D4)
    samplers: dict = field(default_factory=dict)   # sampler name -> texture name


@dataclass
class Mesh:
    vertices: bytes          # packed float32: pos3, normal3, uv2, tangent3 (44 bytes each)
    vertex_count: int
    indices: bytes           # packed uint16
    material: int
    cast_shadows: bool
    visible: bool
    transparent: bool
    renderable: bool = True
    skinned: bool = False


@dataclass
class Node:
    name: str
    kind: int                # 1 = transform, 2 = mesh, 3 = skinned mesh
    active: bool
    matrix: tuple | None = None
    mesh: Mesh | None = None
    children: list = field(default_factory=list)


@dataclass(eq=False)  # hashed by identity
class Kn5:
    version: int
    textures: list
    materials: list
    root: Node


def _read_node(r: Reader) -> Node:
    kind = r.i32()
    name = r.string()
    child_count = r.i32()
    active = r.u8() != 0
    node = Node(name=name, kind=kind, active=active)
    if kind == 1:
        node.matrix = r.f32(16)
    elif kind == 2:
        cast, vis, transp = r.u8(), r.u8(), r.u8()
        vcount = r.i32()
        verts = bytes(r.take(vcount * 44))
        icount = r.i32()
        idx = bytes(r.take(icount * 2))
        mat = r.i32()
        r.i32()           # layer
        r.f32(2)          # lod in / out
        r.f32(4)          # bounding sphere center + radius
        renderable = r.u8() != 0
        node.mesh = Mesh(verts, vcount, idx, mat, bool(cast), bool(vis), bool(transp), renderable)
    elif kind == 3:
        cast, vis, transp = r.u8(), r.u8(), r.u8()
        bones = r.i32()
        for _ in range(bones):
            r.string()
            r.f32(16)
        vcount = r.i32()
        raw = r.take(vcount * 76)  # pos3 normal3 uv2 tangent3 weights4 boneIndices4
        verts = bytearray(vcount * 44)
        for i in range(vcount):
            verts[i * 44:(i + 1) * 44] = raw[i * 76:i * 76 + 44]
        icount = r.i32()
        idx = bytes(r.take(icount * 2))
        mat = r.i32()
        r.i32()           # layer
        r.f32(2)          # lod in / out
        node.mesh = Mesh(bytes(verts), vcount, idx, mat, bool(cast), bool(vis), bool(transp), skinned=True)
    else:
        raise ValueError(f"unknown node class {kind} for '{name}' at {r.p}")
    node.children = [_read_node(r) for _ in range(child_count)]
    return node


def read(path: str, load_textures: bool = True) -> Kn5:
    with open(path, "rb") as f:
        r = Reader(f.read())
    if bytes(r.take(6)) != b"sc6969":
        raise ValueError(f"{path}: not a kn5 file")
    version = r.i32()
    if version > 5:
        r.i32()
    textures = []
    for _ in range(r.i32()):
        r.i32()  # active flag
        name = r.string()
        size = r.i32()
        blob = r.take(size)
        textures.append(Texture(name, bytes(blob) if load_textures else b""))
    materials = []
    for _ in range(r.i32()):
        m = Material(r.string(), r.string(), r.u8(), r.u8() != 0, r.i32())
        for _ in range(r.i32()):
            pname = r.string()
            m.props[pname] = (r.f32(), r.f32(2), r.f32(3), r.f32(4))
        for _ in range(r.i32()):
            sname = r.string()
            r.i32()  # slot
            m.samplers[sname] = r.string()
        materials.append(m)
    root = _read_node(r)
    if r.p != len(r.d):
        raise ValueError(f"{path}: {len(r.d) - r.p} trailing bytes after node tree")
    return Kn5(version, textures, materials, root)


def walk(node: Node, parent=None):
    """Yield (node, parent) depth-first."""
    yield node, parent
    for c in node.children:
        yield from walk(c, node)
