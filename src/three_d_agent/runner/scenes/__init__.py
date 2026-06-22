import re
from pathlib import Path

import mujoco


def load_combined_model(combined_xml: str, asset_mjcf_path: str) -> "mujoco.MjModel":
    """Load a scene-combined MJCF string while resolving mesh files.

    Scenes build their world by injecting floor/table/gripper XML into the
    asset's MJCF and loading the result. The asset MJCF uses ``meshdir="."``
    with relative ``.obj`` filenames; with ``from_xml_string`` MuJoCo resolves
    those against the process CWD (not the asset dir), so mesh assets fail with
    "Error opening file". We collect every referenced mesh file from the asset
    directory and pass them as an in-memory ``assets`` dict so the path resolves
    regardless of CWD.
    """
    asset_dir = Path(asset_mjcf_path).resolve().parent
    assets: dict[str, bytes] = {}
    for fname in re.findall(r'file="([^"]+)"', combined_xml):
        candidate = asset_dir / fname
        if candidate.exists():
            assets[fname] = candidate.read_bytes()
    if assets:
        return mujoco.MjModel.from_xml_string(combined_xml, assets)
    return mujoco.MjModel.from_xml_string(combined_xml)
