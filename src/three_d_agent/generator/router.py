from pathlib import Path
from typing import Optional
from three_d_agent.sad.schema import SAD, PrimitiveKind
from .primitive_builder import PrimitiveBuilder, Builder
from .template_builder import TemplateBuilder
from .mesh_builder import MeshBuilder


def route(sad: SAD) -> tuple[Optional[Builder], str]:
    if any(b.primitive_kind == PrimitiveKind.MESH for b in sad.bodies):
        mb = MeshBuilder()
        ok, reason = mb.supports(sad)
        if ok:
            return mb, ""
        return None, reason

    if sad.joints:
        tb = TemplateBuilder()
        ok, _ = tb.supports(sad)
        if ok:
            return tb, ""

    pb = PrimitiveBuilder()
    ok, _ = pb.supports(sad)
    if ok:
        return pb, ""
    return None, "PrimitiveBuilder does not support this SAD"


def route_and_build(sad: SAD, work_dir: Path) -> tuple[Path, str]:
    builder, reason = route(sad)
    if builder is None:
        raise ValueError(f"No builder available: {reason}")
    return builder.build(sad, work_dir), builder.__class__.__name__
