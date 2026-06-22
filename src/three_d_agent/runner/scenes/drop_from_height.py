"""Drop the asset from a fixed height and check it lands stably.

Unlike gravity_settle (which drops from just above the floor and checks the
velocity settles), this scene drops from a meaningful height and checks the
asset does not tip over or bounce excessively on impact -- a robustness check
for top-heavy or narrow-base assets.
"""
import re
import mujoco
import numpy as np
from three_d_agent.sad.schema import SAD
from three_d_agent.runner.scenes import load_combined_model

DROP_HEIGHT_M = 0.20
SIM_TIME_S = 1.5
TIP_THRESHOLD_RAD = np.deg2rad(20.0)
BOUNCE_THRESHOLD_MPS = 0.05


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


def _wrap_with_floor(asset_xml: str, drop_z: float) -> str:
    floor = (
        '<geom name="drop_floor" type="plane" size="5 5 0.1" '
        'pos="0 0 0" rgba="0.8 0.8 0.8 1"/>'
    )
    body_match = re.search(r'<body\b([^>]*)>', asset_xml)
    if body_match:
        attrs = body_match.group(1)
        pos_match = re.search(r'pos="([^"]+)"', attrs)
        if pos_match:
            px, py, pz = pos_match.group(1).split()
            new_attrs = attrs.replace(
                pos_match.group(0),
                f'pos="{px} {py} {float(pz) + drop_z}"',
            )
        else:
            new_attrs = attrs + f' pos="0 0 {drop_z}"'
        new_body_tag = f'<body{new_attrs}>'
        shifted = asset_xml[: body_match.start()] + new_body_tag + asset_xml[body_match.end():]
    else:
        shifted = asset_xml
    return shifted.replace("<worldbody>", "<worldbody>" + floor, 1)


def _find_root_body(model) -> int:
    for i in range(1, model.nbody):
        if int(model.body_parentid[i]) == 0:
            return i
    return 1


def run(mjcf_path: str, sad: SAD | None = None, sim_time_s: float = SIM_TIME_S) -> dict:
    with open(mjcf_path, "r", encoding="utf-8") as fh:
        asset_xml = fh.read()

    half_h = _asset_half_height(asset_xml)
    drop_z = DROP_HEIGHT_M + half_h

    combined_xml = _wrap_with_floor(asset_xml, drop_z)
    model = load_combined_model(combined_xml, mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = _find_root_body(model)
    initial_pos = np.array(data.xpos[body_id], dtype=float).copy()

    steps = int(sim_time_s / model.opt.timestep)
    max_bounce_vz = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)
        vz = float(data.qvel[2] if data.qvel.ndim > 0 else 0)
        if vz > max_bounce_vz:
            max_bounce_vz = vz

    final_pos = np.array(data.xpos[body_id], dtype=float)
    final_quat = np.array(data.xquat[body_id], dtype=float)
    final_vz = float(data.qvel[2] if data.qvel.ndim > 0 else 0)

    tilt_rad = _tilt_from_quat(final_quat)
    settled = abs(final_vz) < BOUNCE_THRESHOLD_MPS

    violations = []
    if not settled:
        violations.append({
            "kind": "still_bouncing",
            "metric": "final_vertical_velocity_mps",
            "value": final_vz,
            "expected_max": BOUNCE_THRESHOLD_MPS,
            "diagnosis_hint": "asset still bouncing at sim end - "
                              "too bouncy or insufficient damping",
        })
    if tilt_rad > TIP_THRESHOLD_RAD:
        violations.append({
            "kind": "tipped_over",
            "metric": "final_tilt_deg",
            "value": float(np.rad2deg(tilt_rad)),
            "expected_max": float(np.rad2deg(TIP_THRESHOLD_RAD)),
            "diagnosis_hint": "asset tipped > 20deg on impact - "
                              "center of gravity too high or base too narrow",
        })

    return {
        "scene": "drop_from_height",
        "passed": len(violations) == 0,
        "metrics": {
            "drop_height_m": DROP_HEIGHT_M,
            "final_tilt_deg": float(np.rad2deg(tilt_rad)),
            "final_vertical_velocity_mps": final_vz,
            "max_bounce_vz_mps": max_bounce_vz,
            "settled": settled,
            "wall_clock_s": sim_time_s,
        },
        "violations": violations,
    }
