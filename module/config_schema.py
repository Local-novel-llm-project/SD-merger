from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    left: str = Field(min_length=1, validation_alias=AliasChoices("left", "model_a"))
    right: str = Field(min_length=1, validation_alias=AliasChoices("right", "model_b"))
    base_model: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("base_model", "model_c"),
    )
    velocity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("velocity", "alpha"),
    )
    beta: float = Field(default=0.5, ge=0.0, le=1.0)
    strategy: str = "addition"
    target_strategy: Optional[str] = "addition"
    normalization_strategy: str = "none"
    left_right_velocity: float = 1.0
    replace_with: Optional[str] = None
    key_patterns: Optional[List[str] | Dict[str, Any]] = None
    base_alpha: Optional[float] = None
    in_blocks: Optional[List[float]] = None
    mid_block: Optional[List[float]] = None
    out_blocks: Optional[List[float]] = None
    custom_weights: Optional[List[float]] = None
    mbw: Optional[str] = None
    mbw_a: Optional[str] = None
    mbw_b: Optional[str] = None

    @field_validator("strategy", mode="before")
    def validate_strategy(cls, v):
        return v

    @property
    def model_A(self) -> str:
        return self.left

    @property
    def model_B(self) -> str:
        return self.right

    @property
    def model_C(self) -> Optional[str]:
        return self.base_model

    @property
    def alpha(self) -> float:
        return self.velocity


class MergeConfig(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    models: List[ModelConfig] = Field(default_factory=list)
    target_model: Optional[str] = None
    output_dir: str = "output"
    output_name: Optional[str] = None
    save_model: bool = True
    device: str = "cpu"
    dtype: str = "float16"
    lazy_load: bool = True
    bake_in_vae: Optional[str] = None
    extensions: Optional[Dict[str, Any]] = None

    @field_validator("dtype", mode="before")
    def validate_dtype(cls, v):
        allowed = ["float16", "float32", "bfloat16"]
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}, got {v}")
        return v


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str = ""
    negative_prompt: str = ""
    width: int = Field(default=512, gt=0)
    height: int = Field(default=512, gt=0)
    steps: int = Field(default=20, gt=0)
    cfg: float = Field(default=7.0, gt=0.0)
    seed: int = -1
    sampler: str = "euler_ancestral"
    batch_size: int = Field(default=1, gt=0)
