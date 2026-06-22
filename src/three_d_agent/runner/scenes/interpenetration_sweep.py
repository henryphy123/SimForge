import mujoco
from three_d_agent.sad.schema import SAD, JointKind

PENETRATION_THRESHOLD_M = 0.001
DEFAULT_RANGE_M = 0.25
DEFAULT_RANGE_RAD = 1.5708


def run(mjcf_path: str, sad: SAD | None = None, sim_time_s: float = 2.0) -> dict:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    target_joint_id = None
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            target_joint_id = j
            break

    if target_joint_id is None:
        return {
            "scene": "interpenetration_sweep",
            "passed": False,
            "metrics": {},
            "violations": [{
                "kind": "scene_not_applicable",
                "diagnosis_hint": "no hinge or slide joint found to sweep"
            }],
        }

    qpos_adr = model.jnt_qposadr[target_joint_id]
    jtype = model.jnt_type[target_joint_id]

    if jtype == mujoco.mjtJoint.mjJNT_HINGE:
        range_min, range_max = -DEFAULT_RANGE_RAD, DEFAULT_RANGE_RAD
    else:
        range_min, range_max = 0.0, DEFAULT_RANGE_M

    if model.jnt_range is not None and model.jnt_limited[target_joint_id]:
        r = model.jnt_range[target_joint_id]
        range_min = float(r[0])
        range_max = float(r[1])

    if sad:
        for joint in sad.joints:
            if joint.range:
                if jtype == mujoco.mjtJoint.mjJNT_HINGE and joint.kind == JointKind.HINGE:
                    range_min, range_max = joint.range[0], joint.range[1]
                    break
                if jtype == mujoco.mjtJoint.mjJNT_SLIDE and joint.kind == JointKind.PRISMATIC:
                    range_min, range_max = joint.range[0], joint.range[1]
                    break


    steps = int(sim_time_s / model.opt.timestep)
    max_penetration = 0.0

    for step in range(steps + 1):
        frac = step / steps if steps > 0 else 1.0
        target = range_min + (range_max - range_min) * frac
        data.qpos[qpos_adr] = target
        mujoco.mj_forward(model, data)

        for contact in range(data.ncon):
            con = data.contact[contact]
            penetration = con.dist
            if penetration < 0:
                max_penetration = max(max_penetration, -penetration)

    violations = []
    if max_penetration > PENETRATION_THRESHOLD_M:
        violations.append({
            "kind": "interpenetration",
            "metric": "max_penetration_mm",
            "value": max_penetration * 1000,
            "expected_max": PENETRATION_THRESHOLD_M * 1000,
            "diagnosis_hint": "geometry interpenetration during joint sweep"
        })

    return {
        "scene": "interpenetration_sweep",
        "passed": len(violations) == 0,
        "metrics": {
            "max_penetration_mm": max_penetration * 1000,
            "swept_range_min": range_min,
            "swept_range_max": range_max,
            "wall_clock_s": sim_time_s,
        },
        "violations": violations,
    }
