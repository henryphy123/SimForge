import re
import mujoco
import numpy as np
from three_d_agent.sad.schema import SAD
from three_d_agent.runner.scenes import load_combined_model

DEFAULT_ASSET_HALF_HEIGHT = 0.1


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
        elif len(parts) == 1:
            try:
                r = float(parts[0])
                if r > max_half_z:
                    max_half_z = r
            except ValueError:
                continue
    return max_half_z if max_half_z > 0 else DEFAULT_ASSET_HALF_HEIGHT


def _wrap_with_floor(asset_xml: str) -> str:
    """Inject a floor plane and lift the asset root body so it rests on it.

    Builders (PrimitiveBuilder, MeshBuilder) emit only the asset body with a
    freejoint at pos 0 0 0 -- no floor. Without a floor the body freefalls
    forever and gravity_settle can never pass. We add the floor here so the
    scene is self-contained regardless of what the builder produced.
    """
    half_h = _asset_half_height(asset_xml)
    drop_z = half_h + 0.001

    floor = (
        '<geom name="settle_floor" type="plane" size="5 5 0.1" '
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


def run(mjcf_path: str, sad: SAD | None = None, sim_time_s: float = 0.5) -> dict:
    with open(mjcf_path, "r", encoding="utf-8") as fh:
        asset_xml = fh.read()

    combined_xml = _wrap_with_floor(asset_xml)
    model = load_combined_model(combined_xml, mjcf_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    steps = int(sim_time_s / model.opt.timestep)
    settle_threshold = 1e-3
    settle_window = 50
    velocities_squared: list[float] = []

    for _ in range(steps):
        mujoco.mj_step(model, data)
        v = data.qvel
        velocities_squared.append(float(np.dot(v, v)))

    if not np.isfinite(velocities_squared[-1]):
        return {
            "scene": "gravity_settle",
            "passed": False,
            "metrics": {"final_vel_sq": float("nan"), "wall_clock_s": sim_time_s},
            "violations": [{
                "kind": "nan_state",
                "diagnosis_hint": "simulation produced NaN - MJCF physics parameters invalid"
            }],
        }

    window = velocities_squared[-settle_window:]
    settled = all(v < settle_threshold for v in window)
    final_vel_sq = velocities_squared[-1]

    if not settled:
        return {
            "scene": "gravity_settle",
            "passed": False,
            "metrics": {"final_vel_sq": final_vel_sq, "wall_clock_s": sim_time_s},
            "violations": [{
                "kind": "did_not_settle",
                "diagnosis_hint": "body did not settle - likely unstable contact or still bouncing"
            }],
        }

    return {
        "scene": "gravity_settle",
        "passed": True,
        "metrics": {"final_vel_sq": final_vel_sq, "wall_clock_s": sim_time_s},
        "violations": [],
    }
