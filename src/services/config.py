from enum import StrEnum, auto
from typing import List

from pydantic import BaseModel


class Metric(StrEnum):
    COSINE_SIMILARITY = auto()
    EUCLIDEAN_SIMILARITY = auto()
    EUCLIDEAN_DISTANCE = auto()


class PreprocessingConfig(BaseModel):
    # Size threshold
    size_threshold_active: bool = True
    size_threshold_threshold_variable: str = "AbdDiameter"
    size_threshold_lower_bound: float = 30
    size_threshold_upper_bound: float = float("inf")

    # Duplicate detection
    duplicate_detection_active: bool = True
    duplicate_detection_max_source_distance: int = 5
    duplicate_detection_x_coordinate_offset: int = 100
    duplicate_detection_overlap_threshold: float = 0.3
    duplicate_detection_metric: Metric = Metric.COSINE_SIMILARITY
    duplicate_detection_metric_threshold: float = 0.999
    duplicate_detection_similarity_features: List[str] = [
        "EdgeGradient",
        "AbdArea",
        "AvgBlue",
        "AvgGreen",
        "AvgRed",
        "Intensity",
        "Length",
        "Perimeter",
        "Roughness",
        "SigmaIntensity",
        "Transparency",
        "Width",
    ]

    # Air bubble detection
    air_bubble_detection_active: bool = True

    # Biovolume and surface area calculator
    biovol_and_surface_area_calculator_active: bool = True
    '''
    Parameters
    bead_size = The real bead diameter in micrometer
    bead_equiv_diameter = The equivalent diameter of the bead in pixels, calculated from the bead image
    '''
    bead_size: float = 50
    bead_equiv_diameter: float = 30.25

class ClassificationConfig(BaseModel):
    # Object classification
    object_classification_active: bool = True


class Config:
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    classification: ClassificationConfig = ClassificationConfig()
