import shutil
from pathlib import Path

from three_d_agent.sad.schema import SAD, PrimitiveKind
from three_d_agent.generator.mesh import MeshGenerator
from three_d_agent.generator.mesh_cache import get_or_generate
from three_d_agent.generator.convex_decomp import decompose
from three_d_agent.generator.primitive_builder import _geom_size_str, _rgba


class MeshBuilder:
    """Builds an MJCF for SADs containing mesh bodies. Each mesh body resolves a
    .obj via the mesh cache (generating on a miss), copies it into the build dir
    so the asset is self-contained, and emits a mesh geom. When CoACD is
    available, concave meshes are decomposed into multiple convex pieces so
    collision respects concave detail (pockets, hooks, L-shapes); each piece
    becomes its own mesh + geom under the same body. When CoACD is unavailable
    or returns one piece, the mesh collides as its convex hull (Phase 2
    limitation)."""

    def __init__(self, generator: MeshGenerator | None = None):
        self._generator = generator

    def supports(self, sad: SAD) -> tuple[bool, str]:
        if not any(b.primitive_kind == PrimitiveKind.MESH for b in sad.bodies):
            return False, "no mesh bodies"
        if sad.joints:
            return False, "MeshBuilder does not support jointed mesh assets in Phase 2"
        return True, ""

    def _resolve_mesh_pieces(self, body, work_dir: Path) -> list[tuple[str, str]]:
        """Return [(mesh_name, obj_filename), ...] for one mesh body.

        Tries convex decomposition first; on success emits one entry per
        convex piece. Falls back to the single whole-mesh file otherwise.
        Copies every .obj into work_dir.
        """
        key, cached = get_or_generate(
            prompt=body.name, hint=body.mesh_hint, generator=self._generator
        )
        body.mesh_ref = f"cache:{key}"

        pieces_dir = work_dir / "meshes"
        pieces = decompose(cached, pieces_dir)
        if pieces and len(pieces) > 1:
            entries = []
            for i, piece_src in enumerate(pieces):
                obj_name = f"{body.name}_piece{i}.obj"
                shutil.copyfile(piece_src, work_dir / obj_name)
                entries.append((f"{body.name}_mesh_{i}", obj_name))
            return entries

        obj_name = f"{body.name}.obj"
        shutil.copyfile(cached, work_dir / obj_name)
        return [(f"{body.name}_mesh", obj_name)]

    def build(self, sad: SAD, work_dir: Path) -> Path:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "meshes").mkdir(exist_ok=True)

        asset_lines: list[str] = []
        body_lines: list[str] = []

        for b in sad.bodies:
            if b.primitive_kind == PrimitiveKind.MESH:
                pieces = self._resolve_mesh_pieces(b, work_dir)
                for mesh_name, obj_name in pieces:
                    asset_lines.append(f'    <mesh name="{mesh_name}" file="{obj_name}"/>')
                first_mesh_name = pieces[0][0]
                geom_lines = (
                    f'      <geom name="{b.name}_geom_0" type="mesh" '
                    f'mesh="{first_mesh_name}" mass="{b.mass}" rgba="{_rgba(b)}"/>'
                )
                for i, (mesh_name, _) in enumerate(pieces[1:], start=1):
                    geom_lines += (
                        f'\n      <geom name="{b.name}_geom_{i}" type="mesh" '
                        f'mesh="{mesh_name}" mass="0" rgba="{_rgba(b)}"/>'
                    )
                body_lines.append(
                    f'    <body name="{b.name}" pos="0 0 0">\n'
                    f'      <!-- convex decomposition: {len(pieces)} piece(s) -->\n'
                    f'{geom_lines}\n'
                    f'      <freejoint/>\n'
                    f'    </body>'
                )
            else:
                body_lines.append(
                    f'    <body name="{b.name}" pos="0 0 0">\n'
                    f'      <geom name="{b.name}_geom" type="{b.primitive_kind.value}" '
                    f'size="{_geom_size_str(b)}" mass="{b.mass}" rgba="{_rgba(b)}"/>\n'
                    f'      <freejoint/>\n'
                    f'    </body>'
                )

        asset_block = ""
        if asset_lines:
            asset_block = "  <asset>\n" + "\n".join(asset_lines) + "\n  </asset>\n"

        mjcf = (
            f'<mujoco model="{sad.category}">\n'
            f'  <compiler angle="radian" coordinate="local" meshdir="."/>\n'
            f'  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
            f'{asset_block}'
            f'  <worldbody>\n'
            + "\n".join(body_lines) + "\n"
            f'  </worldbody>\n'
            f'</mujoco>\n'
        )
        (work_dir / "asset.mjcf").write_text(mjcf, encoding="utf-8")
        (work_dir / "sad.json").write_text(sad.model_dump_json(indent=2), encoding="utf-8")
        return work_dir
