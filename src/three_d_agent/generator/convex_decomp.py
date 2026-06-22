"""Convex decomposition for concave meshes.

MuJoCo collides mesh geoms against their convex hull by default, so concave
detail (pockets, hooks, L-shapes) is not respected at contact time. This
module uses CoACD to split a mesh into multiple convex pieces; the MJCF
builder then emits one mesh + one geom per piece, and MuJoCo unions their
collision shapes -- recovering concave collision fidelity.

CoACD is an optional dependency: if `import coacd` fails, `decompose` returns
None and the caller falls back to the single-convex-hull path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a minimal .obj into (vertices, triangle-indices).

    Only `v` and `f` lines are parsed. Faces are triangulated naively (fan)
    when they have >3 vertices -- sufficient for the procedural stub output
    and typical CoACD input.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) > 3:
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decompose(obj_path: Path, out_dir: Path, threshold: float = 0.05,
              max_convex_hull: int = -1, mcts_iterations: int = 150) -> list[Path] | None:
    """Split a mesh .obj into convex pieces via CoACD.

    Returns a list of .obj paths (one per convex piece), or None if CoACD is
    unavailable. Single-piece output (convex input) returns a one-element list.
    """
    try:
        import coacd
    except ImportError:
        return None

    coacd.set_log_level("off")
    verts, faces = _read_obj(Path(obj_path))
    mesh = coacd.Mesh()
    mesh.vertices = verts
    mesh.indices = faces

    pieces = coacd.run_coacd(
        mesh, threshold=threshold, max_convex_hull=max_convex_hull,
        mcts_iterations=mcts_iterations, seed=0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(obj_path).stem
    paths: list[Path] = []
    for i, piece in enumerate(pieces):
        p_verts, p_faces = piece[0], piece[1]
        out = out_dir / f"{stem}_piece{i}.obj"
        _write_obj(out, p_verts, p_faces)
        paths.append(out)
    return paths
