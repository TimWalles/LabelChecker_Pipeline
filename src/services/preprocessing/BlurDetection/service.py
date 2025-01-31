from pathlib import Path
from typing import Tuple

import cv2
from numpy.typing import NDArray

from src.enums.LabelChecker import LabelChecker
from src.logging import log_error, log_info
from src.schemas.LabelChecker import LabelCheckerData
from src.utils.image import get_image
from src.utils.ServiceSettings import ServiceSettings, subset_settings

from .config import Config


class BlurDetection:
    config = Config()

    @classmethod
    def process_data(
        cls,
        data: list[LabelCheckerData],
        directory: Path,
        verbose: bool,
        service_settings: ServiceSettings | None,
    ) -> Tuple[list[LabelCheckerData], ServiceSettings | None]:
        if not cls.config.active:
            log_info(
                message=f"[bold magenta]blur detection[/bold magenta] switched [bold red]off[/bold red].",
                verbose=verbose,
            )
            if service_settings:
                service_settings.remove(cls.__name__)
            return data, service_settings

        log_info(
            message=f"""Running [bold magenta]blur detection[/bold magenta] with settings:
        upper threshold value: [bold magenta]{cls.config.threshold_value}[/bold magenta] set to [bold magenta]{LabelChecker.Preprocessing.BLURRY.value}[/bold magenta]
                """,
            verbose=verbose,
        )
        if service_settings:
            service_settings.update({cls.__name__: subset_settings(cls.config.model_dump(), cls.__name__)})
        try:
            return [
                _calculate_laplacian_variance(
                    lc_data=lc_data,
                    directory=directory,
                    threshold_value=cls.config.threshold_value,
                )
                for lc_data in data
            ], service_settings
        except Exception as e:
            log_error(message=f"Error in blur detection: {e}", verbose=verbose)
            return data, service_settings


def _calculate_laplacian_variance(
    lc_data: LabelCheckerData,
    directory: Path,
    threshold_value: float,
) -> LabelCheckerData:
    if lc_data.Preprocessing and not lc_data.Preprocessing in [LabelChecker.Preprocessing.OBJECT.value]:
        return lc_data

    image = _load_image(
        lc_data=lc_data,
        directory=directory,
    )
    score = cv2.Laplacian(image, cv2.CV_64F).var() if image.any() else None
    return _check_conditions(lc_data, score, threshold_value)


def _check_conditions(
    lc_data: LabelCheckerData,
    score: float | None,
    threshold_value: float,
) -> LabelCheckerData:
    if score:
        if not lc_data.Preprocessing or lc_data.Preprocessing in [LabelChecker.Preprocessing.OBJECT.value]:
            if score < threshold_value:
                lc_data.Preprocessing = LabelChecker.Preprocessing.BLURRY.value
                return lc_data

    lc_data.Preprocessing = lc_data.Preprocessing if lc_data.Preprocessing else LabelChecker.Preprocessing.OBJECT.value
    return lc_data


def _load_image(
    lc_data: LabelCheckerData,
    directory: Path,
) -> NDArray:
    file_name, type_image_collage = (lc_data.CollageFile, True) if not lc_data.ImageFilename else (lc_data.ImageFilename, False)
    if not file_name:
        raise ValueError("Image file name not found in data.")

    image_path = directory.joinpath(file_name).as_posix() if type_image_collage else directory.joinpath(directory.name, file_name).as_posix()
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found at path: {image_path}")
    return get_image(
        image_path=image_path,
        type_image_collage=type_image_collage,
        x_loc=lc_data.ImageX,
        y_loc=lc_data.ImageY,
        Width=lc_data.ImageW,
        height=lc_data.ImageH,
    )
