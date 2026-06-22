from pathlib import Path
from three_d_agent.sad.schema import SAD


_JOINT_TYPE_MAP = {
    "hinge": "hinge",
    "prismatic": "slide",
    "free": "free",
    "fixed": None,
}

_MATERIAL_COLORS = {
    "wood": "0.6 0.4 0.2 1",
    "metal": "0.7 0.7 0.7 1",
    "plastic": "0.2 0.6 0.2 1",
    "glass": "0.7 0.8 0.9 0.5",
    "rubber": "0.1 0.1 0.1 1",
}


def _rgba(body) -> str:
    if body.color_rgba:
        return " ".join(str(x) for x in body.color_rgba)
    return _MATERIAL_COLORS[body.material.value]


def _geom_size_str(body) -> str:
    return " ".join(str(s) for s in body.size)


def _render_joint(joint) -> str:
    mjcf_type = _JOINT_TYPE_MAP[joint.kind.value]
    if mjcf_type is None:
        return ""
    axis_str = " ".join(str(a) for a in (joint.axis or [1, 0, 0]))
    range_str = f' range="{joint.range[0]} {joint.range[1]}"' if joint.range else ""
    damping_str = f' damping="{joint.damping}"' if joint.damping else ""
    friction_str = f' frictionloss="{joint.friction_loss}"' if joint.friction_loss else ""
    return (
        f'<joint name="{joint.name}" type="{mjcf_type}" '
        f'axis="{axis_str}"{range_str}{damping_str}{friction_str}/>'
    )


def build(sad: SAD, work_dir: Path) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "meshes").mkdir(exist_ok=True)

    bodies_by_name = {b.name: b for b in sad.bodies}
    joints_by_child = {j.child: j for j in sad.joints}
    children_of: dict[str, list] = {}
    for j in sad.joints:
        children_of.setdefault(j.parent, []).append(j)

    roots = [b for b in sad.bodies if b.name not in joints_by_child]

    def render_body(body, indent: int = 4) -> str:
        pad = " " * indent
        children = children_of.get(body.name, [])
        lines = [f'{pad}<body name="{body.name}" pos="0 0 0">']
        lines.append(
            f'{pad}  <geom name="{body.name}_geom" type="{body.primitive_kind.value}" '
            f'size="{_geom_size_str(body)}" mass="{body.mass}" rgba="{_rgba(body)}"/>'
        )
        for child_joint in children:
            child_body = bodies_by_name.get(child_joint.child)
            if child_body is None:
                continue
            joint_xml = _render_joint(child_joint)
            if joint_xml:
                lines.append(f"{pad}  {joint_xml}")
            lines.append(render_body(child_body, indent + 2))
        lines.append(f'{pad}</body>')
        return "\n".join(lines)

    body_xml = [render_body(root) for root in roots]

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
