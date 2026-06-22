import mujoco
import numpy as np
from three_d_agent.sad.schema import SAD, JointKind

DEFAULT_EXPECTED_MAX_TORQUE_NM = 5.0
DEFAULT_HINGE_RANGE_RAD = np.pi / 2
SWEEP_SPEED_RAD_S = 0.5
PENETRATION_THRESHOLD_M = 0.001


def run(mjcf_path: str, sad: SAD | None = None, sim_time_s: float = 2.0) -> dict:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    hinge_joint_id = None
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_joint_id = j
            break

    if hinge_joint_id is None:
        return {
            "scene": "swing_hinge",
            "passed": False,
            "metrics": {},
            "violations": [{
                "kind": "scene_not_applicable",
                "diagnosis_hint": "no hinge joint found in MJCF"
            }],
        }

    qpos_adr = model.jnt_qposadr[hinge_joint_id]
    dof_adr = model.jnt_dofadr[hinge_joint_id]

    range_min, range_max = -DEFAULT_HINGE_RANGE_RAD, DEFAULT_HINGE_RANGE_RAD
    if sad:
        for joint in sad.joints:
            if joint.kind == JointKind.HINGE and joint.range:
                range_min, range_max = joint.range[0], joint.range[1]
                break
    if model.jnt_range is not None and model.jnt_limited[hinge_joint_id]:
        r = model.jnt_range[hinge_joint_id]
        range_min = float(r[0])
        range_max = float(r[1])

    sweep_span = abs(range_max - range_min)
    sweep_duration = sweep_span / SWEEP_SPEED_RAD_S if SWEEP_SPEED_RAD_S > 0 else sim_time_s

    hinge_body_id = int(model.jnt_bodyid[hinge_joint_id])
    parent_body_id = int(model.body_parentid[hinge_body_id])
    excluded_pair = frozenset((hinge_body_id, parent_body_id))

    steps = int(max(sweep_duration, 0.1) / model.opt.timestep)
    peak_torque = 0.0
    max_penetration = 0.0
    prev_target = float(range_min)

    for step in range(steps + 1):
        frac = step / steps if steps > 0 else 1.0
        target = range_min + (range_max - range_min) * frac
        data.qpos[qpos_adr] = target
        mujoco.mj_forward(model, data)

        if step > 0:
            vel = (target - prev_target) / model.opt.timestep
            torque = abs(model.dof_damping[dof_adr] * vel)
            peak_torque = max(peak_torque, torque)
        prev_target = target

        for contact in range(data.ncon):
            con = data.contact[contact]
            pair = frozenset((
                int(model.geom_bodyid[con.geom1]),
                int(model.geom_bodyid[con.geom2]),
            ))
            if pair == excluded_pair:
                continue
            penetration = con.dist
            if penetration < 0:
                max_penetration = max(max_penetration, -penetration)

    expected_max = DEFAULT_EXPECTED_MAX_TORQUE_NM
    violations = []
    if peak_torque > expected_max:
        violations.append({
            "kind": "torque_exceeds_expected",
            "metric": "peak_actuator_torque_Nm",
            "value": peak_torque,
            "expected_max": expected_max,
            "diagnosis_hint": "hinge mass too high OR hinge damping too high OR panel too long"
        })
    if max_penetration > PENETRATION_THRESHOLD_M:
        violations.append({
            "kind": "interpenetration",
            "metric": "max_penetration_mm",
            "value": max_penetration * 1000,
            "diagnosis_hint": "hinge bodies overlap during sweep"
        })

    return {
        "scene": "swing_hinge",
        "passed": len(violations) == 0,
        "metrics": {
            "peak_actuator_torque_Nm": peak_torque,
            "swept_range_rad": sweep_span,
            "max_penetration_mm": max_penetration * 1000,
            "wall_clock_s": sweep_duration,
        },
        "violations": violations,
    }
