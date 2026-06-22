import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Vec = Tuple[float, float, float]


def _seed(name: str, prompt: str, hint: Optional[Dict[str, Any]]) -> str:
    payload = name + "\0" + prompt + "\0" + json.dumps(hint or {}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _box(hx: float, hy: float, hz: float):
    """Axis-aligned box. MuJoCo takes the convex hull, so this doubles as the
    collision proxy for a 'box' mesh hint."""
    verts = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]
    return verts, faces


def _icosphere(radius: float, subdivisions: int = 1):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts: List[Vec] = [tuple(np.array(v) / np.linalg.norm(v)) for v in base]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    midpoint_cache: Dict[Tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in midpoint_cache:
            return midpoint_cache[key]
        m = (np.array(verts[a]) + np.array(verts[b])) / 2.0
        m = m / np.linalg.norm(m)
        verts.append(tuple(m))
        idx = len(verts) - 1
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    scaled = [tuple(np.array(v) * radius) for v in verts]
    return scaled, [(a + 1, b + 1, c + 1) for a, b, c in faces]


def _cylinder(radius: float, half_len: float, segments: int = 16):
    verts: List[Vec] = []
    for sign in (-1.0, 1.0):
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            verts.append((radius * math.cos(a), radius * math.sin(a), sign * half_len))
    bottom_center = len(verts); verts.append((0.0, 0.0, -half_len))
    top_center = len(verts); verts.append((0.0, 0.0, half_len))

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        bi, bj = i, j
        ti, tj = segments + i, segments + j
        faces.append((bi + 1, bj + 1, tj + 1))
        faces.append((bi + 1, tj + 1, ti + 1))
        faces.append((bottom_center + 1, bj + 1, bi + 1))
        faces.append((top_center + 1, ti + 1, tj + 1))
    return verts, faces


def _write_obj(path: Path, verts, faces, header: str) -> None:
    lines = [f"# {header}"]
    for x, y, z in verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for f in faces:
        lines.append("f " + " ".join(str(i) for i in f))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _size_from_seed(seed: str) -> Tuple[float, float, float]:
    """Deterministic 3-5cm half-extents derived from the prompt hash."""
    out = []
    for k in range(3):
        byte = int(seed[k * 2:k * 2 + 2], 16)
        out.append(0.03 + (byte / 255.0) * 0.02)
    return tuple(out)


class ProceduralStubGenerator:
    """Deterministic procedural mesh generator. Zero new dependencies (numpy
    only). The default Phase 2 backing for the MeshGenerator interface; a real
    image-to-3D generator can replace it behind the same protocol."""

    name = "procedural-stub-v1"

    def generate(self, prompt: str, hint: Optional[Dict[str, Any]] = None) -> Path:
        seed = _seed(self.name, prompt, hint)
        hint = hint or {}
        primitive = hint.get("primitive")
        size = hint.get("size")

        if primitive == "sphere":
            r = float(size[0]) if size else _size_from_seed(seed)[0]
            verts, faces = _icosphere(r)
        elif primitive == "cylinder":
            if size:
                r, h = float(size[0]), float(size[1])
            else:
                s = _size_from_seed(seed)
                r, h = s[0], s[2]
            verts, faces = _cylinder(r, h)
        else:
            if size:
                hx, hy, hz = (float(v) for v in size[:3])
            else:
                hx, hy, hz = _size_from_seed(seed)
            verts, faces = _box(hx, hy, hz)

        out = Path(tempfile.gettempdir()) / f"three_d_agent_stub_{seed}.obj"
        _write_obj(out, verts, faces, header=f"{self.name} {seed}")
        return out
