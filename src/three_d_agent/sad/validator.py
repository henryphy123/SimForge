import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from .schema import (
    SAD, Body, PrimitiveKind, JointKind, Material,
)


def _has_template(category: str) -> bool:
    return importlib.util.find_spec(
        f"three_d_agent.generator.templates.{category}"
    ) is not None


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _body_volume(body: Body) -> float:
    if body.primitive_kind == PrimitiveKind.BOX:
        return 8.0 * body.size[0] * body.size[1] * body.size[2]
    if body.primitive_kind == PrimitiveKind.SPHERE:
        r = body.size[0]
        return (4.0 / 3.0) * math.pi * r ** 3
    if body.primitive_kind in (PrimitiveKind.CYLINDER, PrimitiveKind.CAPSULE):
        r, h = body.size[0], body.size[1]
        return math.pi * r * r * (2.0 * h)
    return 0.0


def validate(sad: SAD, session_dir: Optional[Path] = None) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    body_names = {b.name for b in sad.bodies}

    for body in sad.bodies:
        if body.primitive_kind == PrimitiveKind.MESH:
            if body.mesh_hint is not None and not isinstance(body.mesh_hint, dict):
                errors.append(f"Body '{body.name}': mesh_hint must be an object or null")
            continue

        if body.size is None:
            errors.append(f"Body '{body.name}': {body.primitive_kind.value} requires size")
            continue

        size_len_ok = True
        if body.primitive_kind == PrimitiveKind.BOX and len(body.size) != 3:
            errors.append(f"Body '{body.name}': box needs 3 sizes [hx, hy, hz]")
            size_len_ok = False
        if body.primitive_kind == PrimitiveKind.SPHERE and len(body.size) != 1:
            errors.append(f"Body '{body.name}': sphere needs 1 size [r]")
            size_len_ok = False
        if body.primitive_kind in (PrimitiveKind.CYLINDER, PrimitiveKind.CAPSULE) and len(body.size) != 2:
            errors.append(f"Body '{body.name}': {body.primitive_kind.value} needs 2 sizes [r, half_length]")
            size_len_ok = False

        if not size_len_ok:
            continue

        vol = _body_volume(body)
        if vol > 0 and vol < 1e-6:
            warnings.append(
                f"Body '{body.name}': volume {vol:.2e} m^3 < 1cm^3, possible unit error"
            )

        if vol > 0 and body.material == Material.WOOD:
            density = body.mass / vol
            if density > 5000:
                warnings.append(
                    f"Body '{body.name}': implied density {density:.0f} kg/m^3 too high for wood"
                )

    for joint in sad.joints:
        if joint.kind in (JointKind.HINGE, JointKind.PRISMATIC) and not joint.axis:
            errors.append(f"Joint '{joint.name}': axis required for {joint.kind.value}")
        if joint.range is not None and len(joint.range) != 2:
            errors.append(f"Joint '{joint.name}': range must be [min, max]")
        if joint.damping < 0 or joint.damping > 100:
            warnings.append(f"Joint '{joint.name}': damping {joint.damping} outside [0, 100]")
        if joint.friction_loss < 0:
            warnings.append(f"Joint '{joint.name}': friction_loss {joint.friction_loss} negative")

    if sad.joints and not _has_template(sad.category):
        warnings.append(
            f"Category '{sad.category}' has no template; joints will be ignored "
            f"(bodies built as free-floating primitives)"
        )

    for joint in sad.joints:
        if joint.parent not in body_names:
            errors.append(f"Joint '{joint.name}': parent '{joint.parent}' not in bodies")
        if joint.child not in body_names:
            errors.append(f"Joint '{joint.name}': child '{joint.child}' not in bodies")

    for c in sad.constraints:
        if c.body is not None and c.body not in body_names:
            errors.append(f"Constraint {c.kind.value}: body '{c.body}' not in bodies")
        if c.bodies:
            for b in c.bodies:
                if b not in body_names:
                    errors.append(f"Constraint {c.kind.value}: body '{b}' not in bodies")

    for ref in sad.composition:
        if len(ref.pose) != 7:
            errors.append(
                f"Composition '{ref.asset_ref}': pose must be 7 values "
                f"[x, y, z, qw, qx, qy, qz]"
            )
        if session_dir is not None:
            asset_dir = Path(session_dir) / ref.asset_ref
            if not asset_dir.exists():
                errors.append(
                    f"Composition '{ref.asset_ref}': no such asset in session"
                )
            elif not (asset_dir / f"v{ref.version}" / "asset.mjcf").exists():
                errors.append(
                    f"Composition '{ref.asset_ref}': v{ref.version}/asset.mjcf not built"
                )

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)
