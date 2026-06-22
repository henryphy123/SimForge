import mujoco
from three_d_agent.sad.schema import SAD, JointKind

DEFAULT_EXPECTED_MAX_TORQUE_NM = 5.0
DEFAULT_SLIDE_RANGE_M = 0.25
PENETRATION_THRESHOLD_M = 0.001


def run(mjcf_path: str, sad: SAD | None = None, sim_time_s: float = 2.0) -> dict:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    slide_joint_id = None
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
            slide_joint_id = j
            break

    if slide_joint_id is None:
        return {
            "scene": "pull_drawer",
            "passed": False,
            "metrics": {},
            "violations": [{
                "kind": "no_prismatic_joint",
                "diagnosis_hint": "no prismatic (slide) joint found in MJCF"
            }],
        }

    qpos_adr = model.jnt_qposadr[slide_joint_id]
    dof_adr = model.jnt_dofadr[slide_joint_id]

    drawer_body_id = int(model.jnt_bodyid[slide_joint_id])
    parent_body_id = int(model.body_parentid[drawer_body_id])
    excluded_pair = frozenset((drawer_body_id, parent_body_id))

    range_max = DEFAULT_SLIDE_RANGE_M
    if sad:
        for joint in sad.joints:
            if joint.kind == JointKind.PRISMATIC and joint.range:
                range_max = joint.range[1]
                break

    steps = int(sim_time_s / model.opt.timestep)
    peak_torque = 0.0
    max_penetration = 0.0
    prev_target = 0.0

    for step in range(steps):
        target = range_max * (step / steps)
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

    traveled = range_max

    expected_max = DEFAULT_EXPECTED_MAX_TORQUE_NM
    violations = []
    if peak_torque > expected_max:
        violations.append({
            "kind": "torque_exceeds_expected",
            "metric": "peak_actuator_torque_Nm",
            "value": peak_torque,
            "expected_max": expected_max,
            "diagnosis_hint": "drawer mass too high OR slide friction too high OR drawer size mismatch"
        })
    if max_penetration > PENETRATION_THRESHOLD_M:
        violations.append({
            "kind": "interpenetration",
            "metric": "max_penetration_mm",
            "value": max_penetration * 1000,
            "diagnosis_hint": "drawer and base overlap during sweep"
        })

    return {
        "scene": "pull_drawer",
        "passed": len(violations) == 0,
        "metrics": {
            "peak_actuator_torque_Nm": peak_torque,
            "joint_traveled_m": traveled,
            "max_penetration_mm": max_penetration * 1000,
            "wall_clock_s": sim_time_s,
        },
        "violations": violations,
    }
