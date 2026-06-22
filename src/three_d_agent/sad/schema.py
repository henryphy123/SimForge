from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class PrimitiveKind(str, Enum):
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    MESH = "mesh"


class Material(str, Enum):
    WOOD = "wood"
    METAL = "metal"
    PLASTIC = "plastic"
    GLASS = "glass"
    RUBBER = "rubber"


class JointKind(str, Enum):
    HINGE = "hinge"
    PRISMATIC = "prismatic"
    FREE = "free"
    FIXED = "fixed"


class ConstraintKind(str, Enum):
    GRASPABLE_BY = "graspable_by"
    MUST_STAND_STABLE_ON = "must_stand_stable_on"
    CONTACT_PAIR_REQUIRED = "contact_pair_required"
    MUST_NOT_COLLIDE = "must_not_collide"


class Body(BaseModel):
    name: str
    primitive_kind: PrimitiveKind
    size: Optional[List[float]] = None
    mass: float = Field(gt=0)
    material: Material
    color_rgba: Optional[List[float]] = None
    mesh_ref: Optional[str] = None
    mesh_hint: Optional[Dict[str, Any]] = None


class Joint(BaseModel):
    name: str
    parent: str
    child: str
    kind: JointKind
    axis: Optional[List[float]] = None
    range: Optional[List[float]] = None
    damping: float = 0.0
    friction_loss: float = 0.0


class Constraint(BaseModel):
    kind: ConstraintKind
    body: Optional[str] = None
    bodies: Optional[List[str]] = None
    surface: Optional[str] = None
    gripper_width_range_m: Optional[List[float]] = None
    during_joint: Optional[str] = None


class FeedbackEntry(BaseModel):
    version: int
    source: Literal["human", "sim"]
    msg: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    expected_max: Optional[float] = None


class SourceInputs(BaseModel):
    text: Optional[str] = None
    image_ref: Optional[str] = None
    video_ref: Optional[str] = None


class CompositionRole(str, Enum):
    GRIPPER = "gripper"
    TARGET = "target"


class CompositionRef(BaseModel):
    asset_ref: str
    version: int
    pose: List[float]
    role: CompositionRole


class SAD(BaseModel):
    version: int
    category: str
    source_inputs: SourceInputs
    world_units: Literal["meter"] = "meter"
    bodies: List[Body]
    joints: List[Joint] = []
    constraints: List[Constraint] = []
    composition: List[CompositionRef] = []
    test_scenes: List[str] = []
    feedback_history: List[FeedbackEntry] = []
    open_issues: List[str] = []
