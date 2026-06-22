from pathlib import Path
from typing import Protocol
from three_d_agent.sad.schema import SAD, Body, PrimitiveKind


_MATERIAL_COLORS = {
    "wood": "0.6 0.4 0.2 1",
    "metal": "0.7 0.7 0.7 1",
    "plastic": "0.2 0.6 0.2 1",
    "glass": "0.7 0.8 0.9 0.5",
    "rubber": "0.1 0.1 0.1 1",
}


def _num(x: float) -> str:
    """Render a float compactly: whole numbers as ints (1 not 1.0), others as-is."""
    if float(x).is_integer():
        return str(int(x))
    return str(x)


def _geom_size_str(body: Body) -> str:
    if body.primitive_kind == PrimitiveKind.BOX:
        return f"{_num(body.size[0])} {_num(body.size[1])} {_num(body.size[2])}"
    if body.primitive_kind == PrimitiveKind.SPHERE:
        return f"{_num(body.size[0])}"
    if body.primitive_kind in (PrimitiveKind.CYLINDER, PrimitiveKind.CAPSULE):
        return f"{_num(body.size[0])} {_num(body.size[1])}"
    raise ValueError(f"Unsupported primitive: {body.primitive_kind}")


def _rgba(body: Body) -> str:
    if body.color_rgba:
        return " ".join(_num(x) for x in body.color_rgba)
    return _MATERIAL_COLORS[body.material.value]


class Builder(Protocol):
    def supports(self, sad: SAD) -> tuple[bool, str]: ...
    def build(self, sad: SAD, work_dir: Path) -> Path: ...


class PrimitiveBuilder:
    def supports(self, sad: SAD) -> tuple[bool, str]:
        for b in sad.bodies:
            if b.primitive_kind == PrimitiveKind.MESH:
                return False, "mesh not supported by PrimitiveBuilder"
        return True, ""

    def build(self, sad: SAD, work_dir: Path) -> Path:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "meshes").mkdir(exist_ok=True)

        body_xml: list[str] = []
        for b in sad.bodies:
            body_xml.append(
                f'    <body name="{b.name}" pos="0 0 0">\n'
                f'      <geom name="{b.name}_geom" type="{b.primitive_kind.value}" '
                f'size="{_geom_size_str(b)}" mass="{b.mass}" rgba="{_rgba(b)}"/>\n'
                f'      <freejoint/>\n'
                f'    </body>'
            )

        mjcf = (
            f'<mujoco model="{sad.category}">\n'
            f'  <compiler angle="radian" coordinate="local"/>\n'
            f'  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
            f'  <worldbody>\n'
            + "\n".join(body_xml) + "\n"
            f'  </worldbody>\n'
            f'</mujoco>\n'
        )
        (work_dir / "asset.mjcf").write_text(mjcf, encoding="utf-8")
        (work_dir / "sad.json").write_text(sad.model_dump_json(indent=2), encoding="utf-8")
        return work_dir
