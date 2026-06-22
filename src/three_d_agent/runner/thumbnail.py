import mujoco
import numpy as np
from PIL import Image

_VIEWS = (
    (90.0, -10.0),
    (0.0, -10.0),
    (90.0, -89.0),
    (45.0, -30.0),
)


def render_thumbnail(mjcf_path: str, out_path: str, tile_w: int = 240, tile_h: int = 180) -> str:
    """Render four camera views of the asset and composite them into a 2x2 PNG.

    Each view frames the whole model via MuJoCo's default free camera (which
    centers on the model and backs off by its extent), then overrides only the
    orbit angles so the asset is seen from front/side/top/iso.
    """
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=tile_h, width=tile_w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    base_distance = cam.distance

    tiles = []
    try:
        for azimuth, elevation in _VIEWS:
            cam.azimuth = azimuth
            cam.elevation = elevation
            cam.distance = base_distance
            renderer.update_scene(data, camera=cam)
            tiles.append(renderer.render())
    finally:
        renderer.close()

    top = np.hstack([tiles[0], tiles[1]])
    bottom = np.hstack([tiles[2], tiles[3]])
    grid = np.vstack([top, bottom])
    Image.fromarray(grid).save(out_path)
    return str(out_path)
