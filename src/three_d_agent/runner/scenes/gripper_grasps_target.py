"""Cross-asset scene: a parallel gripper grasps a (mesh) target and lifts it.

The scene reads SAD `composition` to find the gripper and target asset refs in
the same session. The target asset (typically a mesh) is loaded from its built
MJCF so the real mesh geometry and its convex-hull collision are exercised.
The gripper is loaded from its built MJCF (produced by the
simple_parallel_gripper template, which emits a base + two offset fingers with
position actuators), wrapped in a lift carriage so the whole gripper can rise,
and driven closed then lifted.
"""
import re
from pathlib import Path

import mujoco
import numpy as np

from three_d_agent.sad.schema import SAD, CompositionRole

LIFT_DELTA_M = 0.05
SLIP_THRESHOLD_M = 0.02
PENETRATION_THRESHOLD_M = 0.003
GRIP_OVERTRAVEL_M = 0.004
LIFT_KP = 2000.0
CLOSE_DURATION_S = 0.3
LIFT_DURATION_S = 0.6
HOLD_DURATION_S = 0.3
TABLE_HALF_HEIGHT = 0.05


def _session_dir(mjcf_path: str) -> Path:
    return Path(mjcf_path).resolve().parents[2]


def _resolve_ref(session_dir: Path, asset_ref: str, version: int) -> Path | None:
    candidate = session_dir / asset_ref / f"v{version}" / "asset.mjcf"
    return candidate if candidate.exists() else None


def _pick_refs(sad: SAD | None):
    gripper = target = None
    if sad:
        for ref in sad.composition:
            if ref.role == CompositionRole.GRIPPER:
                gripper = ref
            elif ref.role == CompositionRole.TARGET:
                target = ref
    return gripper, target


def _mesh_half_extents(model, geom_id: int) -> tuple[float, float]:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        s = model.geom_size[geom_id]
        return float(s[0]) or 0.025, float(s[2]) or 0.025
    adr = int(model.mesh_vertadr[mesh_id])
    num = int(model.mesh_vertnum[mesh_id])
    verts = model.mesh_vert[adr:adr + num].reshape(-1, 3)
    half_x = float(np.max(np.abs(verts[:, 0])))
    half_z = float(np.max(np.abs(verts[:, 2])))
    return max(half_x, 1e-3), max(half_z, 1e-3)


def _fail(kind: str, hint: str) -> dict:
    return {
        "scene": "gripper_grasps_target",
        "passed": False,
        "metrics": {},
        "violations": [{"kind": kind, "diagnosis_hint": hint}],
    }


def _extract_worldbody_inner(gripper_xml: str) -> tuple[str, str]:
    """Return (inner_xml, actuator_xml) from a gripper MJCF.

    inner_xml is the content between <worldbody> and </worldbody>; we wrap it
    in a lift_carriage body. actuator_xml is the gripper's own <actuator>...</actuator>
    block (finger position servos), which we merge with the lift actuator.

    Collision is disabled on every non-finger gripper geom (the base, which is
    wide enough to envelop a small target, and any structural geom). Only the
    finger geoms should touch the target; otherwise the base shoves it away.
    """
    wb_match = re.search(r'<worldbody>(.*?)</worldbody>', gripper_xml, re.DOTALL)
    inner = wb_match.group(1).strip() if wb_match else ""
    inner = _disable_non_finger_collision(inner)
    act_match = re.search(r'<actuator>(.*?)</actuator>', gripper_xml, re.DOTALL)
    actuator_inner = act_match.group(1).strip() if act_match else ""
    return inner, actuator_inner


def _disable_non_finger_collision(inner_xml: str) -> str:
    """Add contype=0 conaffinity=0 to every <geom> whose name is not a finger.

    Finger geoms (name contains "finger") keep their collision so they can
    grip; the base and any other structural geom are made non-colliding.
    """
    def repl(m: re.Match) -> str:
        tag = m.group(0)
        name_m = re.search(r'name="([^"]*)"', tag)
        name = name_m.group(1) if name_m else ""
        if "finger" in name or "contype" in tag:
            return tag
        return tag[:-2] + ' contype="0" conaffinity="0"/>' if tag.endswith("/>") else tag

    return re.sub(r'<geom\b[^>]*/>', repl, inner_xml)


def run(mjcf_path: str, sad: SAD | None = None) -> dict:
    session_dir = _session_dir(mjcf_path)
    gripper_ref, target_ref = _pick_refs(sad)

    if target_ref is not None:
        resolved = _resolve_ref(session_dir, target_ref.asset_ref, target_ref.version)
        if resolved is None:
            return _fail("dangling_ref",
                         f"target asset '{target_ref.asset_ref}' v{target_ref.version} "
                         f"not built in session")
        target_mjcf = resolved
        target_pose = target_ref.pose
    else:
        target_mjcf = Path(mjcf_path)
        target_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    if gripper_ref is None:
        return _fail("dangling_ref",
                     "gripper_grasps_target requires a 'gripper' role in composition")
    gripper_mjcf = _resolve_ref(session_dir, gripper_ref.asset_ref, gripper_ref.version)
    if gripper_mjcf is None:
        return _fail("dangling_ref",
                     f"gripper asset '{gripper_ref.asset_ref}' v{gripper_ref.version} "
                     f"not built in session")

    target_dir = target_mjcf.parent
    target_xml = target_mjcf.read_text(encoding="utf-8")
    gripper_xml = gripper_mjcf.read_text(encoding="utf-8")

    probe = mujoco.MjModel.from_xml_path(str(target_mjcf))
    probe_geom = -1
    for g in range(probe.ngeom):
        if int(probe.geom_bodyid[g]) >= 1:
            probe_geom = g
            break
    if probe_geom < 0:
        return _fail("scene_not_applicable", "target asset has no body geom")
    half_x, half_z = _mesh_half_extents(probe, probe_geom)

    tx, ty = float(target_pose[0]), float(target_pose[1])
    table_top = TABLE_HALF_HEIGHT
    target_z = table_top + half_z + 0.001
    started_above_table = target_z >= table_top
    carriage_z = target_z

    floor = (
        '<geom name="floor" type="plane" size="2 2 0.1" pos="0 0 0" '
        'rgba="0.8 0.8 0.8 1"/>'
    )
    table = (
        f'<body name="table" pos="{tx} {ty} {table_top - TABLE_HALF_HEIGHT}">'
        f'<geom name="table_top" type="box" size="0.4 0.4 {TABLE_HALF_HEIGHT}" '
        f'mass="50" rgba="0.5 0.5 0.5 1"/></body>'
    )

    gripper_inner, gripper_actuators = _extract_worldbody_inner(gripper_xml)
    gripper_body = (
        f'<body name="lift_carriage" pos="{tx} {ty} {carriage_z}">'
        f'  <joint name="lift" type="slide" axis="0 0 1" '
        f'range="-0.001 {LIFT_DELTA_M + 0.1}" damping="5.0"/>'
        f'  <geom name="carriage_geom" type="box" size="0.012 0.012 0.006" '
        f'mass="0.2" contype="0" conaffinity="0"/>'
        f'  {gripper_inner}'
        f'</body>'
    )
    actuators = (
        f'<actuator>'
        f'  {gripper_actuators}'
        f'  <position name="act_lift" joint="lift" kp="{LIFT_KP}"/>'
        f'</actuator>'
    )

    def _shift_target(xml: str) -> str:
        m = re.search(r'<body\b([^>]*)>', xml)
        if not m:
            return xml
        attrs = m.group(1)
        pos_m = re.search(r'pos="([^"]+)"', attrs)
        new_pos = f'pos="{tx} {ty} {target_z}"'
        if pos_m:
            new_attrs = attrs.replace(pos_m.group(0), new_pos)
        else:
            new_attrs = attrs + " " + new_pos
        return xml[:m.start()] + f'<body{new_attrs}>' + xml[m.end():]

    combined = _shift_target(target_xml)
    combined = combined.replace(
        "<worldbody>", "<worldbody>" + floor + table + gripper_body, 1
    )
    combined = combined.replace("</mujoco>", actuators + "</mujoco>", 1)

    scene_file = target_dir / "_grasp_scene.xml"
    scene_file.write_text(combined, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_file))
    except (ValueError, RuntimeError) as exc:
        return _fail(
            "scene_not_applicable",
            f"could not compose gripper + target into one scene: {exc}. "
            "The gripper and target must be distinct assets (the gripper a "
            "simple_parallel_gripper).",
        )
    finally:
        scene_file.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    fl_id = bid("finger_left")
    fr_id = bid("finger_right")
    lift_id = bid("lift_carriage")
    table_id = bid("table")
    reserved = {fl_id, fr_id, lift_id, table_id, 0}
    base_id = bid("base")
    if base_id > 0:
        reserved.add(base_id)
    target_id = next(i for i in range(1, model.nbody) if i not in reserved)

    act_l = aid("act_slide_left")
    act_r = aid("act_slide_right")
    act_lift = aid("act_lift")

    if fl_id < 0 or fr_id < 0 or act_l < 0 or act_r < 0:
        return _fail(
            "scene_not_applicable",
            "gripper ref is not a parallel gripper: expected finger_left/"
            "finger_right bodies with act_slide_left/act_slide_right actuators "
            "(build a simple_parallel_gripper asset for the 'gripper' role)",
        )

    lift_qadr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift")])

    def peak_grasp_force_and_penetration() -> tuple[float, float]:
        force = 0.0
        pen = 0.0
        f6 = np.zeros(6)
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])}
            if target_id in pair and (fl_id in pair or fr_id in pair):
                mujoco.mj_contactForce(model, data, ci, f6)
                force = max(force, float(np.linalg.norm(f6[:3])))
                pen = max(pen, -float(con.dist))
        return force, pen

    contact_established = False
    peak_penetration = 0.0

    def _finger_geom(body_id: int) -> int:
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) == body_id:
                return g
        return -1

    def _close_cmd(finger_body_id: int) -> float:
        geom = _finger_geom(finger_body_id)
        half_thickness = float(model.geom_size[geom][0]) if geom >= 0 else 0.0
        rest_inner_gap = abs(float(data.xpos[finger_body_id][0]) - tx) - half_thickness
        desired_gap = max(0.0, half_x - GRIP_OVERTRAVEL_M)
        return max(0.0, rest_inner_gap - desired_gap)

    close_l = _close_cmd(fl_id)
    close_r = _close_cmd(fr_id)

    original_gravity = np.array(model.opt.gravity, dtype=float).copy()
    model.opt.gravity[:] = 0.0
    close_steps = int(CLOSE_DURATION_S / model.opt.timestep)
    for step in range(close_steps):
        frac = (step + 1) / close_steps
        data.ctrl[act_l] = close_l * frac
        data.ctrl[act_r] = close_r * frac
        data.ctrl[act_lift] = 0.0
        mujoco.mj_step(model, data)
        f, pen = peak_grasp_force_and_penetration()
        if f > 0.0:
            contact_established = True
        peak_penetration = max(peak_penetration, pen)

    model.opt.gravity[:] = original_gravity
    target_start_z = float(data.xpos[target_id][2])
    carriage_start_z = float(data.qpos[lift_qadr])

    lift_steps = int(LIFT_DURATION_S / model.opt.timestep)
    for step in range(lift_steps):
        frac = (step + 1) / lift_steps
        data.ctrl[act_l] = close_l
        data.ctrl[act_r] = close_r
        data.ctrl[act_lift] = LIFT_DELTA_M * frac
        mujoco.mj_step(model, data)
        f, pen = peak_grasp_force_and_penetration()
        if f > 0.0:
            contact_established = True
        peak_penetration = max(peak_penetration, pen)

    hold_steps = int(HOLD_DURATION_S / model.opt.timestep)
    for _ in range(hold_steps):
        data.ctrl[act_l] = close_l
        data.ctrl[act_r] = close_r
        data.ctrl[act_lift] = LIFT_DELTA_M
        mujoco.mj_step(model, data)

    target_final_z = float(data.xpos[target_id][2])
    carriage_final_z = float(data.qpos[lift_qadr])
    target_rise = target_final_z - target_start_z
    carriage_rise = carriage_final_z - carriage_start_z
    lag = carriage_rise - target_rise

    violations = []
    if not started_above_table:
        violations.append({
            "kind": "miss_table",
            "metric": "target_start_z_m",
            "value": target_start_z,
            "expected_max": None,
            "diagnosis_hint": "target started below the table surface",
        })
    if not contact_established:
        violations.append({
            "kind": "no_grasp",
            "metric": "target_rise_mm",
            "value": target_rise * 1000.0,
            "expected_max": None,
            "diagnosis_hint": "gripper fingers never contacted the target",
        })
    if peak_penetration > PENETRATION_THRESHOLD_M:
        violations.append({
            "kind": "interpenetration",
            "metric": "penetration_mm",
            "value": peak_penetration * 1000.0,
            "expected_max": PENETRATION_THRESHOLD_M * 1000.0,
            "diagnosis_hint": "gripper and target overlapped during close - "
                              "fingers commanded past the target surface",
        })
    if contact_established and lag > SLIP_THRESHOLD_M:
        violations.append({
            "kind": "slip",
            "metric": "lag_mm",
            "value": lag * 1000.0,
            "expected_max": SLIP_THRESHOLD_M * 1000.0,
            "diagnosis_hint": "target slipped out of the grasp during lift - "
                              "grip force or friction too low",
        })

    return {
        "scene": "gripper_grasps_target",
        "passed": len(violations) == 0,
        "metrics": {
            "target_rise_mm": target_rise * 1000.0,
            "carriage_rise_mm": carriage_rise * 1000.0,
            "lag_mm": lag * 1000.0,
            "penetration_mm": peak_penetration * 1000.0,
            "contact_established": contact_established,
        },
        "violations": violations,
    }
