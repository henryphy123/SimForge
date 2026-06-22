import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from three_d_agent.generator.mesh import MeshGenerator
from three_d_agent.generator.mesh.stub import ProceduralStubGenerator


def _root() -> Path:
    env = os.environ.get("THREE_D_AGENT_ROOT")
    return Path(env) if env else Path.home() / ".3d-agent"


def cache_key(generator_name: str, prompt: str, hint: Optional[Dict[str, Any]]) -> str:
    payload = generator_name + "\0" + prompt + "\0" + json.dumps(hint or {}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(key: str, root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else _root()
    return base / "mesh_cache" / f"{key}.obj"


def get_or_generate(
    prompt: str,
    hint: Optional[Dict[str, Any]] = None,
    generator: Optional[MeshGenerator] = None,
    root: Optional[Path] = None,
) -> tuple[str, Path]:
    """Resolve a mesh for (generator, prompt, hint), generating on a cache miss.

    Returns (key, path) where path points at the cached .obj. Deterministic:
    the same inputs always yield the same key and identical bytes.
    """
    generator = generator or ProceduralStubGenerator()
    key = cache_key(generator.name, prompt, hint)
    dest = cache_path(key, root)
    if dest.exists():
        return key, dest

    produced = generator.generate(prompt, hint)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, dest)
    return key, dest
