from enum import StrEnum, auto


class LabelChecker:
    class Preprocessing(StrEnum):
        OBJECT = auto()
        SMALL = auto()
        LARGE = auto()
        BLURRY = auto()
        BUBBLE = auto()
        DUPLICATE = auto()
