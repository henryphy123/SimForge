"""Shake the asset on a vibrating table and check it stays put.

The table oscillates vertically for a fixed duration; the asset must not tip
excessively or slide off. This complements nudge_robustness (single lateral
pulse) by exercising sustained dynamic disturbance across the whole body.
"""
import re
import mujoco
import numpy as np
from three_d_agent.sad.schema import SAD
from three_d_agent.runner.scenes import load_combined_model

TABLE_HALF_HEIGHT = 0.05
SHAKE_AMPLITUDE_M = 0.01
SHAKE_FREQUENCY_HZ = 15.0
SHAKE_DURATION_S = 1.5
SETTLE_DURATION_S = 0.5
TIP_THRESHOLD_RAD = np.deg2rad(20.0)
SLIDE_THRESHOLD_M = 0.02


def _asset_half_height(asset_xml: str) -> float:
    sizes = re.findall(r'size="([^"]+)"', asset_xml)
    max_half_z = 0.0
    for s in sizes:
        parts = s.split()
        if len(parts) >= 3:
            try:
                half_z = float(parts[2])
                if half_z > max_half_z:
                    max_half_z = half_z
            except ValueError:
                continue
        elif len(parts) == 1:
            try:
                r = float(parts[0])
                if r > max_half_z:
                    max_half_z = r
            except ValueError:
                continue
    return max_half_z if max_half_z > 0 else 0.05


def _tilt_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat
    rx = 2.0 * (x * z + w * y)
    ry = 2.0 * (y * z - w * x)
    rz = 1.0 - 2.0 * (x * x + y * y)
    up = np.array([rx, ry, rz])
    up = up / (np.linalg.norm(up) + 1e-12)
    cos_t = float(np.clip(up[2], -1.0, 1.0))
    return float(np.arccos(cos_t))


def _wrap_with_shaking_table(asset_xml: str, rest_z: float) -> str:
    """Inject a table body with a slide joint so we can drive it vertically."""
    table_body = (
        f'<body name="shake_table" pos="0 0 {TABLE_HALF_HEIGHT}">'
        f'  <joint name="table_lift" type="slide" axis="0 0 1" '
        f'range="{-SHAKE_AMPLITUDE_M} {SHAKE_AMPLITUDE_M}" damping="0.0"/>'
        f'  <geom name="table_top" type="box" size="0.4 0.4 {TABLE_HALF_HEIGHT}" '
        f'mass="20.0" rgba="0.5 0.5 0.5 1"/>'
        f'</body>'
    )
    actuator = (
        f'<actuator>'
        f'  <position name="table_act" joint="table_lift" kp="10000.0"/>'
        f'</actuator>'
    )

    body_match = re.search(r'<body\b([^>]*)>', asset_xml)
    if body_match:
        attrs = body_match.group(1)
        pos_match = re.search(r'pos="([^"]+)"', attrs)
        if pos_match:
            px, py, pz = pos_match.group(1).split()
            new_attrs = attrs.replace(
                pos_match.group(0),
                f'pos="{px} {py} {float(pz) + rest_z}"',
            )
        else:
            new_attrs = attrs + f' pos="0 0 {rest_z}"'
        new_body_tag = f'<body{new_attrs}>'
        shifted = asset_xml[: body_match.start()] + new_body_tag + asset_xml[body_match.end():]
    else:
        shifted = asset_xml

    out = shifted.replace("<worldbody>", "<worldbody>" + table_body, 1)
    out = out.replace("</mujoco>", actuator + "</mujoco>", 1)
    return out


def _find_root_body(model, table_id: int) -> int:
    for i in range(1, model.nbody):
        if i == table_id:
            continue
        if int(model.body_parentid[i]) == 0:
            return i
    return 1


def run(mjcf_path: str, sad: SAD | None = None) -> dict:
    with open(mjcf_path, "r", encoding="utf-8") as fh:
        asset_xml = fh.read()

    half_h = _asset_half_height(asset_xml)
    asset_z = TABLE_HALF_HEIGHT + half_h + 0.001

    combined_xml = _wrap_with_shaking_table(asset_xml, asset_z)
    model = load_combined_model(combined_xml, mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shake_table")
    body_id = _find_root_body(model, table_id)
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "table_act")

    initial_pos = np.array(data.xpos[body_id], dtype=float).copy()

    shake_steps = int(SHAKE_DURATION_S / model.opt.timestep)
    omega = 2.0 * np.pi * SHAKE_FREQUENCY_HZ
    for step in range(shake_steps):
        t = step * model.opt.timestep
        data.ctrl[act_id] = SHAKE_AMPLITUDE_M * np.sin(omega * t)
        mujoco.mj_step(model, data)

    settle_steps = int(SETTLE_DURATION_S / model.opt.timestep)
    for _ in range(settle_steps):
        data.ctrl[act_id] = 0.0
        mujoco.mj_step(model, data)

    final_pos = np.array(data.xpos[body_id], dtype=float)
    final_quat = np.array(data.xquat[body_id], dtype=float)

    tilt_rad = _tilt_from_quat(final_quat)
    slide_m = float(np.hypot(final_pos[0] - initial_pos[0],
                             final_pos[1] - initial_pos[1]))

    violations = []
    if tilt_rad > TIP_THRESHOLD_RAD:
        violations.append({
            "kind": "tipped_over",
            "metric": "final_tilt_deg",
            "value": float(np.rad2deg(tilt_rad)),
            "expected_max": float(np.rad2deg(TIP_THRESHOLD_RAD)),
            "diagnosis_hint": "asset tipped > 20deg during shake - "
                              "center of gravity too high",
        })
    if slide_m > SLIDE_THRESHOLD_M:
        violations.append({
            "kind": "slid_off",
            "metric": "slide_distance_mm",
            "value": slide_m * 1000.0,
            "expected_max": SLIDE_THRESHOLD_M * 1000.0,
            "diagnosis_hint": "asset slid > 2cm during shake - "
                              "friction too low or base too small",
        })

    return {
        "scene": "shake_test",
        "passed": len(violations) == 0,
        "metrics": {
            "final_tilt_deg": float(np.rad2deg(tilt_rad)),
            "slide_distance_mm": slide_m * 1000.0,
            "shake_amplitude_mm": SHAKE_AMPLITUDE_M * 1000.0,
            "shake_frequency_hz": SHAKE_FREQUENCY_HZ,
            "wall_clock_s": SHAKE_DURATION_S + SETTLE_DURATION_S,
        },
        "violations": violations,
    }
