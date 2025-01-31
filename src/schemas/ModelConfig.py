from pydantic import BaseModel, Field
from typing import List


class ModelConfig(BaseModel):
    Name: str
    Version: str = "1"
    Framework: str
    Class_names: List[str]
    Input_shape: List[int]
    Features: List[str]

    class Config:
        # Allow setting fields using dictionary keys
        arbitrary_types_allowed = True

    def __init__(self, model_config: dict, **kwargs) -> None:
        super().__init__(**model_config, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ModelConfig(\n"
            f"Name={self.Name},\n"
            f"Version={self.Version},\n"
            f"Framework={self.Framework},\n"
            f"Class_names={self.Class_names},\n"
            f"Input_shape={self.Input_shape},\n"
            f"Features={self.Features}\n)"
        )
