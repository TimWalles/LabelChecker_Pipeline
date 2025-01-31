from pydantic import BaseModel


class Config(BaseModel):
    active: bool = False
    threshold_value: float = 0
