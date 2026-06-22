import mujoco
import numpy as np
from three_d_agent.sad.schema import SAD

NUDGE_FORCE_N = 1.0
PULSE_DURATION_S = 0.05
SETTLE_DURATION_S = 1.0
TIP_THRESHOLD_RAD = np.deg2rad(30.0)


def _tilt_from_quat(quat: np.ndarray) -> float:
    """Angle (rad) between body's local up-axis and world z."""
    w, x, y, z = quat
    rx = 2.0 * (x * z + w * y)
    ry = 2.0 * (y * z - w * x)
    rz = 1.0 - 2.0 * (x * x + y * y)
    up = np.array([rx, ry, rz])
    up = up / (np.linalg.norm(up) + 1e-12)
    cos_t = float(np.clip(up[2], -1.0, 1.0))
    return float(np.arccos(cos_t))


def _find_root_body(model) -> int:
    """First non-world body in the model (the asset's root)."""
    for i in range(1, model.nbody):
        if int(model.body_parentid[i]) == 0:
            return i
    return 1


def run(mjcf_path: str, sad: SAD | None = None) -> dict:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = _find_root_body(model)

    body_half_height = 0.05
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == body_id:
            sz = model.geom_size[g]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and sz[2] > 0:
                body_half_height = max(body_half_height, float(sz[2]))
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER and sz[1] > 0:
                body_half_height = max(body_half_height, float(sz[1]))
    moment_arm = body_half_height

    pulse_steps = max(1, int(PULSE_DURATION_S / model.opt.timestep))
    settle_steps = int(SETTLE_DURATION_S / model.opt.timestep)

    directions = [
        ("+x", np.array([NUDGE_FORCE_N, 0.0, 0.0]),
         np.array([0.0, NUDGE_FORCE_N * moment_arm, 0.0])),
        ("-x", np.array([-NUDGE_FORCE_N, 0.0, 0.0]),
         np.array([0.0, -NUDGE_FORCE_N * moment_arm, 0.0])),
        ("+y", np.array([0.0, NUDGE_FORCE_N, 0.0]),
         np.array([-NUDGE_FORCE_N * moment_arm, 0.0, 0.0])),
        ("-y", np.array([0.0, -NUDGE_FORCE_N, 0.0]),
         np.array([NUDGE_FORCE_N * moment_arm, 0.0, 0.0])),
    ]

    per_dir_tilt = {}
    max_tilt_rad = 0.0
    worst_dir = None

    for label, force, torque in directions:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        data.xfrc_applied[body_id, 0:3] = force
        data.xfrc_applied[body_id, 3:6] = torque
        for _ in range(pulse_steps):
            mujoco.mj_step(model, data)
        data.xfrc_applied[body_id, 0:6] = 0.0
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)

        tilt = _tilt_from_quat(np.array(data.xquat[body_id]))
        per_dir_tilt[label] = tilt
        if tilt > max_tilt_rad:
            max_tilt_rad = tilt
            worst_dir = label

    violations = []
    if max_tilt_rad > TIP_THRESHOLD_RAD:
        violations.append({
            "kind": "tips_over",
            "metric": "max_tilt_deg",
            "value": float(np.rad2deg(max_tilt_rad)),
            "expected_max": 30.0,
            "diagnosis_hint": (
                f"asset tips > 30deg when nudged along {worst_dir} - "
                "center of gravity too high or base too narrow"
            )
        })

    return {
        "scene": "nudge_robustness",
        "passed": len(violations) == 0,
        "metrics": {
            "max_tilt_deg": float(np.rad2deg(max_tilt_rad)),
            "worst_direction": worst_dir,
            "per_direction_tilt_deg": {k: float(np.rad2deg(v)) for k, v in per_dir_tilt.items()},
        },
        "violations": violations,
    }
