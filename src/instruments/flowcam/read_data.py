from pathlib import Path
from typing import Tuple

import pandas as pd

from src.error.handler import Tracker, error_handler

from .normalizers.normalizer import normalize_flowcam_data


def read_data(
    file_path: Path,
    running_classification: bool,
    tracker: Tracker,
) -> Tuple[pd.DataFrame, Tracker]:
    data, tracker = read_flowcam_data(
        file_path=file_path,
        running_classification=running_classification,
        tracker=tracker,
    )
    return data, tracker


def read_flowcam_data(
    file_path: Path,
    running_classification: bool,
    tracker: Tracker,
) -> Tuple[pd.DataFrame, Tracker]:

    data = pd.read_csv(file_path, engine="pyarrow", encoding="latin1")
    if data.empty:
        tracker = error_handler(tracker=tracker, name=file_path.name, desc="Failed to load data.")
    else:
        reprocess = file_path.name.startswith("LabelChecker_")
        data = normalize_flowcam_data(
            data=data,
            sample_name=file_path.parent.name,
            running_classification=running_classification,
            reprocess=reprocess,
        )
        tracker.successful += 1
    return data, tracker
