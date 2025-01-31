from pathlib import Path
from pydantic import BaseModel, field_validator


class ModelInfo(BaseModel):
    name: str
    version: str = "1"
    file_path: Path
    config_file: Path

    class Config:
        arbitrary_types_allowed = True

    @field_validator("file_path", "config_file")
    def path_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    def __repr__(self) -> str:
        return (
            f"ModelInfo(\n"
            f"Name={self.name},\n"
            f"Version={self.version},\n"
            f"ModelPath={self.file_path},\n"
            f"ConfigFile={self.config_file}\n)"
        )
