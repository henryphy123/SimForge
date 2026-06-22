"""Simple parallel gripper template.

Unlike the generic _common.build (which places every body at the origin),
this emits a functional gripper: a base in the center, two fingers that rest
at their open position, and position actuators that drive the fingers closed.

Kinematic convention (so the asset can actually grip):
- Each finger body rests at its OPEN position, offset to one side by
  ``open + finger_half_thickness`` so the inner faces sit ``open`` from center.
- The slide joint axis points INWARD (toward center).
- The joint range is ``[0, open]``: ``qpos == 0`` is the open rest position,
  ``qpos == open`` brings the inner faces together at the base center (closed).

A position actuator per finger lets a driver command any opening in
``[0, open]``; commanding ``open`` closes the grip.
"""
from pathlib import Path

from three_d_agent.sad.schema import SAD

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


def _finger_offset(joint) -> float:
    """How far from base center the finger sits at rest (open position).

    Uses the joint's max range; defaults to 0.05m if no range is set.
    """
    if joint.range and len(joint.range) >= 2:
        return float(joint.range[1])
    return 0.05


def build(sad: SAD, work_dir: Path) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "meshes").mkdir(exist_ok=True)

    bodies = {b.name: b for b in sad.bodies}
    joints_by_child = {j.child: j for j in sad.joints}

    base_body = next(b for b in sad.bodies if b.name not in joints_by_child)
    base_geom = (
        f'    <body name="{base_body.name}" pos="0 0 0">\n'
        f'      <geom name="{base_body.name}_geom" type="{base_body.primitive_kind.value}" '
        f'size="{_geom_size_str(base_body)}" mass="{base_body.mass}" '
        f'rgba="{_rgba(base_body)}"/>\n'
    )

    finger_lines: list[str] = []
    actuator_lines: list[str] = []
    for j in sad.joints:
        finger = bodies.get(j.child)
        if finger is None:
            continue
        declared = j.axis or [1, 0, 0]
        side = -1.0 if declared[0] < 0 else 1.0
        in_axis = [-side, 0.0, 0.0]
        open_gap = _finger_offset(j)
        finger_half_thickness = finger.size[0] if finger.size else 0.0
        fx = side * (open_gap + finger_half_thickness)
        damping = f' damping="{j.damping}"' if j.damping else ""
        friction = f' frictionloss="{j.friction_loss}"' if j.friction_loss else ""
        finger_lines.append(
            f'      <body name="{finger.name}" pos="{fx:.4f} 0 0">\n'
            f'        <joint name="{j.name}" type="slide" '
            f'axis="{in_axis[0]} {in_axis[1]} {in_axis[2]}" '
            f'range="0 {open_gap}"{damping}{friction}/>\n'
            f'        <geom name="{finger.name}_geom" type="{finger.primitive_kind.value}" '
            f'size="{_geom_size_str(finger)}" mass="{finger.mass}" '
            f'rgba="{_rgba(finger)}"/>\n'
            f'      </body>'
        )
        actuator_lines.append(
            f'  <position name="act_{j.name}" joint="{j.name}" kp="4000.0" '
            f'ctrlrange="0 {open_gap}"/>'
        )

    body_xml = base_geom + "\n".join(finger_lines) + "\n    </body>"
    actuator_block = ""
    if actuator_lines:
        actuator_block = "  <actuator>\n" + "\n".join(actuator_lines) + "\n  </actuator>\n"

    mjcf = (
        f'<mujoco model="{sad.category}">\n'
        f'  <compiler angle="radian" coordinate="local"/>\n'
        f'  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
        f'  <worldbody>\n'
        f'{body_xml}\n'
        f'  </worldbody>\n'
        f'{actuator_block}'
        f'</mujoco>\n'
    )
    (work_dir / "asset.mjcf").write_text(mjcf, encoding="utf-8")
    (work_dir / "sad.json").write_text(sad.model_dump_json(indent=2), encoding="utf-8")
    return work_dir
