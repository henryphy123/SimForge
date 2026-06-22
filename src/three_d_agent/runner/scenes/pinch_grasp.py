import mujoco
import numpy as np
from three_d_agent.sad.schema import SAD, ConstraintKind
from three_d_agent.runner.scenes import load_combined_model

GRASP_FORCE_MIN_N = 1.0
CLOSE_DURATION_S = 0.1
HOLD_DURATION_S = 1.0
SLIP_THRESHOLD_M = 0.1
FINGER_HALF_THICKNESS = 0.005
FINGER_CLEARANCE = 0.001
SQUEEZE_OVERTRAVEL = 0.02
ACTUATOR_KP = 4000.0


def _target_extents(asset_model, target_id: int) -> tuple[float, float]:
    """Return (half_x, half_z) of the target body from its geoms.

    Falls back to a small default when the body has no box-like geom.
    """
    half_x = 0.0
    half_z = 0.0
    for g in range(asset_model.ngeom):
        if int(asset_model.geom_bodyid[g]) != target_id:
            continue
        half_x = max(half_x, float(asset_model.geom_size[g][0]))
        half_z = max(half_z, float(asset_model.geom_size[g][2]))
    if half_x <= 0.0:
        half_x = 0.025
    if half_z <= 0.0:
        half_z = 0.025
    return half_x, half_z


def _wrap_with_gripper(asset_xml: str, target_body_name: str | None,
                       mjcf_path: str) -> tuple[str, float]:
    """Build a combined MJCF: the asset plus a world-anchored parallel gripper
    whose position-actuated fingers close along x onto the target body.

    Returns (combined_xml, close_target). The gripper is fixed to the world at
    the target's resting position so that, once it grips, friction holds the
    (free) target against gravity.
    """
    asset_model = load_combined_model(asset_xml, mjcf_path)
    asset_data = mujoco.MjData(asset_model)
    mujoco.mj_forward(asset_model, asset_data)

    target_id = -1
    if target_body_name:
        target_id = mujoco.mj_name2id(
            asset_model, mujoco.mjtObj.mjOBJ_BODY, target_body_name
        )
    if target_id < 0:
        target_id = 1 if asset_model.nbody > 1 else 0
    target_pos = np.array(asset_data.xpos[target_id], dtype=float)
    half_x, half_z = _target_extents(asset_model, target_id)

    gx, gy, gz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    finger_start = half_x + FINGER_HALF_THICKNESS + FINGER_CLEARANCE
    finger_half_z = max(half_z, 0.02)
    close_target = FINGER_CLEARANCE + SQUEEZE_OVERTRAVEL

    gripper_xml = (
        f'<body name="gripper_base" pos="{gx} {gy} {gz}">'
        f'  <geom name="gbase_geom" type="box" size="0.01 0.01 0.01" mass="0.3" contype="0" conaffinity="0"/>'
        f'  <body name="finger_left" pos="-{finger_start} 0 0">'
        f'    <joint name="slide_l" type="slide" axis="1 0 0" range="-0.001 {finger_start + 0.02}" damping="2.0"/>'
        f'    <geom name="fl_geom" type="box" size="{FINGER_HALF_THICKNESS} 0.01 {finger_half_z}" '
        f'mass="0.05" friction="0.02 0.005 0.0001"/>'
        f'  </body>'
        f'  <body name="finger_right" pos="{finger_start} 0 0">'
        f'    <joint name="slide_r" type="slide" axis="-1 0 0" range="-0.001 {finger_start + 0.02}" damping="2.0"/>'
        f'    <geom name="fr_geom" type="box" size="{FINGER_HALF_THICKNESS} 0.01 {finger_half_z}" '
        f'mass="0.05" friction="0.02 0.005 0.0001"/>'
        f'  </body>'
        f'</body>'
    )
    actuator_xml = (
        f'<actuator>'
        f'  <position name="act_l" joint="slide_l" kp="{ACTUATOR_KP}"/>'
        f'  <position name="act_r" joint="slide_r" kp="{ACTUATOR_KP}"/>'
        f'</actuator>'
    )

    combined = asset_xml.replace("<worldbody>", "<worldbody>" + gripper_xml, 1)
    combined = combined.replace("</mujoco>", actuator_xml + "</mujoco>", 1)
    return combined, close_target


def _grasp_target_body_name(sad: SAD | None) -> str | None:
    if not sad:
        return None
    for c in sad.constraints:
        if c.kind == ConstraintKind.GRASPABLE_BY and c.body:
            return c.body
    return None


def _contact_force(model, data, ci) -> float:
    f = np.zeros(6)
    mujoco.mj_contactForce(model, data, ci, f)
    return float(np.linalg.norm(f[:3]))


def run(mjcf_path: str, sad: SAD | None = None) -> dict:
    with open(mjcf_path, "r", encoding="utf-8") as fh:
        asset_xml = fh.read()

    target_body_name = _grasp_target_body_name(sad)
    combined_xml, close_target = _wrap_with_gripper(asset_xml, target_body_name, mjcf_path)

    model = load_combined_model(combined_xml, mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    target_id = -1
    if target_body_name:
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    if target_id < 0:
        for i in range(1, model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if nm and (nm.startswith("finger") or nm == "gripper_base"):
                continue
            target_id = i
            break

    if target_id < 0:
        return {
            "scene": "pinch_grasp",
            "passed": False,
            "metrics": {},
            "violations": [{
                "kind": "scene_not_applicable",
                "diagnosis_hint": "no target body to grasp",
            }],
        }

    finger_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_left")
    finger_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger_right")
    act_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_l")
    act_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_r")

    def peak_grasp_force() -> float:
        best = 0.0
        for ci in range(data.ncon):
            con = data.contact[ci]
            pair = {int(model.geom_bodyid[con.geom1]), int(model.geom_bodyid[con.geom2])}
            if target_id in pair and (finger_l_id in pair or finger_r_id in pair):
                best = max(best, _contact_force(model, data, ci))
        return best

    grasp_force_N = 0.0
    contact_established = False

    original_gravity = np.array(model.opt.gravity, dtype=float).copy()

    model.opt.gravity[:] = 0.0
    close_steps = int(CLOSE_DURATION_S / model.opt.timestep)
    for step in range(close_steps):
        frac = (step + 1) / close_steps
        data.ctrl[act_l] = close_target * frac
        data.ctrl[act_r] = close_target * frac
        mujoco.mj_step(model, data)
        f = peak_grasp_force()
        if f > 0.0:
            contact_established = True
            grasp_force_N = max(grasp_force_N, f)

    model.opt.gravity[:] = original_gravity
    target_initial_pos = np.array(data.xpos[target_id], dtype=float).copy()
    hold_steps = int(HOLD_DURATION_S / model.opt.timestep)
    for _ in range(hold_steps):
        data.ctrl[act_l] = close_target
        data.ctrl[act_r] = close_target
        mujoco.mj_step(model, data)
        f = peak_grasp_force()
        if f > 0.0:
            contact_established = True
            grasp_force_N = max(grasp_force_N, f)

    target_final_pos = np.array(data.xpos[target_id], dtype=float)
    slip_m = float(np.linalg.norm(target_final_pos - target_initial_pos))

    violations = []
    if not contact_established:
        violations.append({
            "kind": "no_contact",
            "metric": "grasp_force_N",
            "value": grasp_force_N,
            "expected_max": None,
            "diagnosis_hint": "gripper fingers never contact target - "
                              "handle unreachable or gripper mispositioned",
        })
    elif grasp_force_N < GRASP_FORCE_MIN_N:
        violations.append({
            "kind": "force_too_low",
            "metric": "grasp_force_N",
            "value": grasp_force_N,
            "expected_max": None,
            "diagnosis_hint": "grasp force < 1N - handle too thin or too slippery",
        })
    if contact_established and slip_m > SLIP_THRESHOLD_M:
        violations.append({
            "kind": "slipped",
            "metric": "slip_mm",
            "value": slip_m * 1000,
            "expected_max": SLIP_THRESHOLD_M * 1000,
            "diagnosis_hint": "target slipped out of grasp - "
                              "surface too slippery or grip force too low",
        })

    return {
        "scene": "pinch_grasp",
        "passed": len(violations) == 0,
        "metrics": {
            "grasp_force_N": grasp_force_N,
            "contact_established": contact_established,
            "slip_mm": slip_m * 1000,
        },
        "violations": violations,
    }
