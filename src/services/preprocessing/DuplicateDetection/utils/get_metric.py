from typing import Callable

import numpy as np

from src.services.config import Config, Metric, PreprocessingConfig


def get_metric(
    config: PreprocessingConfig,
) -> Callable:
    match config.duplicate_detection_metric:
        case Metric.COSINE_SIMILARITY:
            return cosine_similarity
        case Metric.EUCLIDEAN_SIMILARITY:
            return euclidean_similarity
        case Metric.EUCLIDEAN_DISTANCE:
            return euclidean_distance
        case _:
            raise ValueError(f"Unsupported metric: {config.metric}")


def cosine_similarity(v1, v2) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def euclidean_similarity(v1, v2) -> float:
    return 1 / (1 + np.linalg.norm(v1 - v2))


def euclidean_distance(v1, v2) -> float:
    return np.linalg.norm(v1 - v2)
