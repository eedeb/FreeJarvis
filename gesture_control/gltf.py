"""A reader for binary glTF, enough to get a skinned mesh out of one.

Not a general glTF implementation. It reads the parts of the format the glove
uses -- geometry, per-vertex skin bindings, the joint list and flat material
colours -- and ignores animation, cameras, textures, morph targets and the
extension registry.

The format makes this easy in a way FBX does not. A .glb is a twelve byte
header and then chunks: one of JSON describing the scene, one of raw binary
holding every array. An accessor names a slice of that binary and says how to
read it, so there is no parser to write, only indexing.
"""

from __future__ import annotations

import json
import pathlib
import struct

import numpy as np

_MAGIC = 0x46546C67          # "glTF"
_JSON, _BIN = 0x4E4F534A, 0x004E4942
_COMPONENT = {5120: "<i1", 5121: "<u1", 5122: "<i2",
              5123: "<u2", 5125: "<u4", 5126: "<f4"}
# The largest value of each integer component type, for reading a normalised
# attribute back as a fraction. Weights are usually floats, but the format
# allows them packed as bytes or shorts and Blender will emit that.
_FULL = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
          "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GLTF:
    """A parsed .glb: the JSON tree, the binary blob, and accessor reads."""

    def __init__(self, path: pathlib.Path | str):
        raw = pathlib.Path(path).read_bytes()
        magic, _version, _length = struct.unpack_from("<III", raw, 0)
        if magic != _MAGIC:
            raise ValueError(f"{path} is not a binary glTF")
        self.json: dict = {}
        self.bin = b""
        pos = 12
        while pos + 8 <= len(raw):
            size, kind = struct.unpack_from("<II", raw, pos)
            body = raw[pos + 8:pos + 8 + size]
            if kind == _JSON:
                self.json = json.loads(body)
            elif kind == _BIN:
                self.bin = body
            pos += 8 + size + (-size % 4)
        if not self.json:
            raise ValueError(f"{path} has no JSON chunk")

    def accessor(self, index: int) -> np.ndarray:
        """One accessor, as (count, width) -- or (count,) for scalars."""
        acc = self.json["accessors"][index]
        width = _WIDTH[acc["type"]]
        dtype = np.dtype(_COMPONENT[acc["componentType"]])
        count = acc["count"]
        if "bufferView" not in acc:                  # allowed, and means zeros
            return np.zeros((count, width), np.float32)

        view = self.json["bufferViews"][acc["bufferView"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        packed = dtype.itemsize * width
        stride = view.get("byteStride") or packed
        if stride == packed:
            out = np.frombuffer(self.bin, dtype, count * width,
                                start).reshape(count, width)
        else:
            # Interleaved. Walk the buffer as bytes and pick out the columns,
            # which costs one copy and keeps the read vectorised.
            span = np.frombuffer(self.bin, np.uint8, count * stride, start)
            out = span.reshape(count, stride)[:, :packed].copy()
            out = out.view(dtype).reshape(count, width)
        if acc.get("normalized") and acc["componentType"] in _FULL:
            out = out.astype(np.float32) / _FULL[acc["componentType"]]
        return out.reshape(count) if width == 1 else out


def _srgb(linear) -> float:
    """glTF colours are linear; the renderer paints in sRGB."""
    c = float(np.clip(linear, 0.0, 1.0))
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def load_skinned(path: pathlib.Path | str) -> dict:
    """The first skinned mesh in the file, flattened into plain arrays.

    Returns the vertices and normals in rest space, one material index per
    triangle, and everything needed to pose it: which joints each vertex
    follows and how strongly, the joint names, and where each joint sits at
    rest.

    Primitives are concatenated. glTF splits a mesh at every material boundary
    -- it cannot share a vertex between two of them -- so a five-material glove
    arrives as five separate vertex arrays that have to be stitched back into
    one before anything can be done with it.
    """
    gltf = GLTF(path)
    tree = gltf.json
    skins = tree.get("skins") or []
    if not skins:
        raise ValueError(f"{path} has no skin -- it is not a rigged model")
    skin = skins[0]

    node_of = {}
    for index, node in enumerate(tree.get("nodes", [])):
        if "mesh" in node and "skin" in node:
            node_of.setdefault("mesh", index)
    mesh_index = None
    for node in tree.get("nodes", []):
        if "mesh" in node and node.get("skin") == 0:
            mesh_index = node["mesh"]
            break
    if mesh_index is None:
        mesh_index = 0

    verts, norms, faces, joints, weights, material = [], [], [], [], [], []
    # ...and a colour per vertex as well as per face. The split glTF forces on
    # a multi-material mesh is useful exactly once: because no vertex is shared
    # between two materials, every vertex has an unambiguous colour, so a
    # renderer that wants one per vertex does not have to duplicate anything.
    vcolour = []
    base = 0
    for prim in tree["meshes"][mesh_index]["primitives"]:
        attr = prim["attributes"]
        position = gltf.accessor(attr["POSITION"]).astype(np.float64)
        normal = (gltf.accessor(attr["NORMAL"]).astype(np.float64)
                  if "NORMAL" in attr else np.zeros_like(position))
        bind = gltf.accessor(attr["JOINTS_0"]).astype(np.int32)
        share = gltf.accessor(attr["WEIGHTS_0"]).astype(np.float64)
        index = (gltf.accessor(prim["indices"]).astype(np.int64)
                 if "indices" in prim else np.arange(len(position)))
        verts.append(position)
        norms.append(normal)
        joints.append(bind)
        weights.append(share)
        tri = index.reshape(-1, 3) + base
        faces.append(tri)
        material.append(np.full(len(tri), prim.get("material", 0), np.int32))
        vcolour.append(np.full(len(position), prim.get("material", 0), np.int32))
        base += len(position)

    weights = np.concatenate(weights)
    total = weights.sum(axis=1, keepdims=True)
    weights = weights / np.where(total > 1e-9, total, 1.0)

    # Where each joint sits at rest. The inverse bind matrix takes a vertex
    # from rest space into that joint's space, so its inverse is the joint's
    # own rest placement, and the translation of that is the joint's position.
    # Reading it back out of the file this way means the app never has to
    # carry a copy of the rest pose the model was built in.
    inverse = gltf.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4)
    inverse = np.transpose(inverse, (0, 2, 1))       # glTF stores columns
    rest = np.linalg.inv(inverse)

    colours = []
    for mat in tree.get("materials", []):
        pbr = mat.get("pbrMetallicRoughness", {})
        rgba = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
        colours.append([_srgb(c) * 255.0 for c in rgba[2::-1]])   # to BGR

    palette = (np.array(colours, dtype=float) if colours
               else np.array([[200.0, 200.0, 200.0]]))
    slot = np.concatenate(vcolour)
    return {
        "vcolour": palette[np.clip(slot, 0, len(palette) - 1)],
        "verts": np.concatenate(verts),
        "normals": np.concatenate(norms),
        "faces": np.concatenate(faces),
        "material": np.concatenate(material),
        "joints": np.concatenate(joints),
        "weights": weights,
        "joint_names": [tree["nodes"][n].get("name", f"joint{n}")
                        for n in skin["joints"]],
        "rest": rest,
        "colours": palette,
    }
