from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class MeshGenerator(Protocol):
    """Produces a .obj mesh from a prompt.

    Contract:
    - Deterministic: same (name, prompt, hint) -> same output bytes.
    - Writes exactly one .obj file and returns its path; the caller owns caching.
    - No network, no GPU, no large model weights (Phase-2-compatible on this box).
    """

    name: str

    def generate(self, prompt: str, hint: Optional[Dict[str, Any]] = None) -> Path:
        ...
