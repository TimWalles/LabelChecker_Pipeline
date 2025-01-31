from pathlib import Path
from typing import Tuple

from src.enums.Instruments import Instruments
from src.error.handler import Tracker
from src.schemas.LabelChecker import LabelCheckerData
from src.normalizers.LabelChecker import LabelCheckerNormalizer
from src.logging import log_error
from .flowcam.read_data import read_data as read_flowcam_data


def read_data(
    file_path: Path,
    instrument: Instruments,
    tracker: Tracker,
    running_classification: bool = False,
    normalizer: LabelCheckerNormalizer = LabelCheckerNormalizer(),
) -> Tuple[list[LabelCheckerData], Tracker]:
    match instrument:
        case Instruments.FLOWCAM.value:
            data, tracker = read_flowcam_data(
                file_path=file_path,
                running_classification=running_classification,
                tracker=tracker,
            )
        case _:
            log_error(
                f"Unknown instrument name. Know instrument names are {[i.value for i in Instruments]}. Exiting script"
            )
            exit()
    return normalizer.normalize_to_lc_df(data), tracker
