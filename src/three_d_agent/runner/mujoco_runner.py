from importlib import import_module
from three_d_agent.sad.schema import SAD


def run_scene(scene_name: str, mjcf_path: str, sad: SAD | None = None) -> dict:
    module = import_module(f"three_d_agent.runner.scenes.{scene_name}")
    return module.run(mjcf_path=mjcf_path, sad=sad)
