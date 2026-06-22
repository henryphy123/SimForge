import re
import mujoco
import numpy as np
from three_d_agent.sad.schema import SAD
from three_d_agent.runner.scenes import load_combined_model

TILT_THRESHOLD_RAD = np.deg2rad(5.0)
SLIDE_THRESHOLD_M = 0.01
DEFAULT_ASSET_HALF_HEIGHT = 0.1
TABLE_HALF_HEIGHT = 0.05


def _asset_half_height(asset_xml: str) -> float:
    """Estimate the asset's vertical half-extent from box geom sizes in the MJCF."""
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
    return max_half_z if max_half_z > 0 else DEFAULT_ASSET_HALF_HEIGHT


def _wrap_with_table(asset_xml: str) -> str:
    """Build a combined MJCF: a table box at z=0 plus the asset placed on top.

    Strategy: inject a table body into worldbody, then shift the asset's root
    body `pos` upward so it rests on the table. We locate the first <body ...>
    tag inside worldbody and rewrite its pos to lift it by `drop_z`. The asset
    keeps its own joints (incl. freejoint, which stays top-level under world).
    """
    half_h = _asset_half_height(asset_xml)
    drop_z = TABLE_HALF_HEIGHT + half_h + 0.001

    table_body = (
        f'<body name="table" pos="0 0 0">'
        f'<geom name="table_top" type="box" size="1.0 1.0 {TABLE_HALF_HEIGHT}" '
        f'mass="20.0" rgba="0.5 0.5 0.5 1"/>'
        f'</body>'
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

    wrapped = shifted.replace("<worldbody>", "<worldbody>" + table_body, 1)
    return wrapped


def _tilt_from_quat(quat: np.ndarray) -> float:
    """Angle (rad) between the body's local up-axis and world z, from quaternion."""
    w, x, y, z = quat
    rx = 2.0 * (x * z + w * y)
    ry = 2.0 * (y * z - w * x)
    rz = 1.0 - 2.0 * (x * x + y * y)
    up = np.array([rx, ry, rz])
    up = up / (np.linalg.norm(up) + 1e-12)
    cos_t = float(np.clip(up[2], -1.0, 1.0))
    return float(np.arccos(cos_t))


def run(mjcf_path: str, sad: SAD | None = None, sim_time_s: float = 2.0) -> dict:
    with open(mjcf_path, "r", encoding="utf-8") as fh:
        asset_xml = fh.read()

    combined_xml = _wrap_with_table(asset_xml)
    model = load_combined_model(combined_xml, mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    asset_body_id = -1
    for i in range(1, model.nbody):
        if i == table_id:
            continue
        asset_body_id = i
        break
    if asset_body_id < 0:
        asset_body_id = 1

    initial_pos = np.array(data.xpos[asset_body_id], dtype=float).copy()
    initial_quat = np.array(data.xquat[asset_body_id], dtype=float).copy()

    steps = int(sim_time_s / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)

    final_pos = np.array(data.xpos[asset_body_id], dtype=float)
    final_quat = np.array(data.xquat[asset_body_id], dtype=float)

    tilt_rad = _tilt_from_quat(final_quat)
    slide_m = float(np.hypot(final_pos[0] - initial_pos[0], final_pos[1] - initial_pos[1]))

    violations = []
    if tilt_rad > TILT_THRESHOLD_RAD:
        violations.append({
            "kind": "unstable_tilt",
            "metric": "final_tilt_deg",
            "value": float(np.rad2deg(tilt_rad)),
            "expected_max": 5.0,
            "diagnosis_hint": "asset tips over on the table - center of gravity too high or base too narrow"
        })
    if slide_m > SLIDE_THRESHOLD_M:
        violations.append({
            "kind": "unstable_slide",
            "metric": "slide_distance_mm",
            "value": slide_m * 1000.0,
            "expected_max": SLIDE_THRESHOLD_M * 1000.0,
            "diagnosis_hint": "asset slides off the table - friction too low or contact patch too small"
        })

    return {
        "scene": "place_on_table",
        "passed": len(violations) == 0,
        "metrics": {
            "final_tilt_deg": float(np.rad2deg(tilt_rad)),
            "slide_distance_mm": slide_m * 1000.0,
            "wall_clock_s": sim_time_s,
        },
        "violations": violations,
    }
