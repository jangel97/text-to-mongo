from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FieldRole(str, Enum):
    identifier = "identifier"
    measure = "measure"
    timestamp = "timestamp"
    category = "category"
    text = "text"
    enum = "enum"
    boolean = "boolean"


class FieldDef(BaseModel):
    name: str
    type: str  # MongoDB type string: "string", "int", "double", "date", "bool", "objectId", "array", "object"
    role: FieldRole
    description: str = ""
    enum_values: list[str] | None = None


class SchemaDef(BaseModel):
    collection: str
    fields: list[FieldDef]
    domain: str

    @property
    def field_names(self) -> set[str]:
        return {f.name for f in self.fields}

    def fields_by_role(self, role: FieldRole) -> list[FieldDef]:
        return [f for f in self.fields if f.role == role]


class AllowedOps(BaseModel):
    stage_operators: list[str] = Field(default_factory=list)
    expression_operators: list[str] = Field(default_factory=list)

    @property
    def all_operators(self) -> list[str]:
        return self.stage_operators + self.expression_operators


class TrainingExample(BaseModel):
    schema_def: SchemaDef = Field(alias="schema")
    allowed_ops: AllowedOps
    intent: str
    output: dict[str, Any]
    is_negative: bool = False

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Evaluation result models
# ---------------------------------------------------------------------------

class SyntaxResult(BaseModel):
    valid_json: bool = False
    has_type: bool = False
    type_value: str | None = None
    has_body: bool = False  # pipeline or filter present
    pipeline_well_formed: bool = False  # each stage is a dict with one $-key (aggregate only)
    passed: bool = False
    errors: list[str] = Field(default_factory=list)


class OperatorResult(BaseModel):
    used_operators: set[str] = Field(default_factory=set)
    violations: set[str] = Field(default_factory=set)  # used but not in allowed list
    unsafe_operators: set[str] = Field(default_factory=set)  # from hard blocklist
    passed: bool = False

    model_config = {"arbitrary_types_allowed": True}


class FieldResult(BaseModel):
    referenced_fields: set[str] = Field(default_factory=set)
    hallucinated_fields: set[str] = Field(default_factory=set)
    coverage: float = 0.0  # fraction of schema fields referenced
    passed: bool = False

    model_config = {"arbitrary_types_allowed": True}


class GeneralizationResult(BaseModel):
    train_syntax_pass_rate: float = 0.0
    held_out_syntax_pass_rate: float = 0.0
    train_operator_pass_rate: float = 0.0
    held_out_operator_pass_rate: float = 0.0
    train_field_pass_rate: float = 0.0
    held_out_field_pass_rate: float = 0.0
    gaps: dict[str, float] = Field(default_factory=dict)  # metric_name -> gap
    flagged: bool = False  # True if any gap > 5%


class EvalResult(BaseModel):
    example: TrainingExample
    prediction: str
    syntax: SyntaxResult
    operators: OperatorResult
    fields: FieldResult
    passed_all: bool = False


class EvalReport(BaseModel):
    results: list[EvalResult]
    total: int = 0
    syntax_pass_rate: float = 0.0
    operator_pass_rate: float = 0.0
    field_pass_rate: float = 0.0
    overall_pass_rate: float = 0.0
    generalization: GeneralizationResult | None = None
