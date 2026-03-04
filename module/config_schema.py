from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Required fields for merging
    model_A: str = Field(
        alias="model_a", default=""
    )  # Alias logic handling can be complex, better to allow flexibility
    model_B: str = Field(alias="model_b", default="")
    model_C: str = Field(alias="model_c", default="")

    # Generic model reference (some strategies just use "models": ["a", "b"])
    # We will relax strict required fields if 'target_strategy' etc handles it

    # Algorithm parameters
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    beta: float = Field(default=0.5, ge=0.0, le=1.0)

    # Strategy
    strategy: str = "addition"
    target_strategy: Optional[str] = None

    # MBW specific
    base_alpha: Optional[float] = None
    in_blocks: Optional[List[float]] = None
    mid_block: Optional[List[float]] = None  # Sometimes a list of 1 element
    out_blocks: Optional[List[float]] = None
    custom_weights: Optional[List[float]] = None

    @field_validator("strategy", mode="before")
    def validate_strategy(cls, v):
        # We can add a strict list of allowed strategies here later if needed
        return v


class MergeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Top level configuration
    models: List[Union[ModelConfig, Dict[str, Any]]] = Field(default_factory=list)
    output_dir: str = "output"
    device: str = "cpu"
    dtype: str = "float16"

    # Optional extensions/hooks configurations
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
    sampler: str = "euler_a"
    batch_size: int = Field(default=1, gt=0)
