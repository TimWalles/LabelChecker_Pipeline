from pathlib import Path

from src.logging import log_info
from src.schemas.LabelChecker import LabelCheckerData
from src.utils.ServiceSettings import ServiceSettings

from .classification.ObjectClassification.service import ObjectClassification
from .preprocessing.AirBubbleDetection.service import AirBubbleDetection
from .preprocessing.BlurDetection.service import BlurDetection
from .preprocessing.DetritusDetection.service import DetritusDetection
from .preprocessing.DuplicateDetection.service import DuplicateDetection
from .preprocessing.SizeThreshold.service import SizeThreshold


class DataPreprocessor:
    # Initialize services that require models and prevent re-loading of same model each time
    air_bubble_detection = AirBubbleDetection()
    detritus_detection = DetritusDetection()

    def process_label_checker_data(
        self,
        data: list[LabelCheckerData],
        data_directory: Path,
        verbose: bool,
        save_settings: bool,
    ) -> list[LabelCheckerData]:
        service_settings = ServiceSettings(directory=data_directory) if save_settings else None  # only initialize if save_settings is set to true

        data, service_settings = SizeThreshold.process_data(
            data=data,
            verbose=verbose,
            service_settings=service_settings,
        )

        data, service_settings = BlurDetection.process_data(
            data=data,
            directory=data_directory,
            verbose=verbose,
            service_settings=service_settings,
        )

        data, service_settings = self.air_bubble_detection.process_data(
            data=data,
            data_directory=data_directory,
            verbose=verbose,
            service_settings=service_settings,
        )

        data, service_settings = DuplicateDetection.process_data(
            data=data,
            verbose=verbose,
            service_settings=service_settings,
        )

        data, service_settings = self.detritus_detection.process_data(
            data=data,
            verbose=verbose,
            service_settings=service_settings,
        )
        if service_settings:
            log_info(message=f"Service setting file [bold magenta]{service_settings.FilePath.name}[/bold magenta] saved")
            service_settings.save()
        return data


class DataClassifier:
    # Initialize services that require models and prevent re-loading of same model each time
    object_classification = ObjectClassification()

    def classify_label_checker_data(
        self,
        data: list[LabelCheckerData],
        data_directory: Path,
        verbose: bool,
        save_settings: bool,
    ) -> list[LabelCheckerData]:
        service_settings = ServiceSettings(directory=data_directory) if save_settings else None  # only initialize if save_settings is set to true

        data, service_settings = self.object_classification.process_data(
            data=data,
            data_directory=data_directory,
            verbose=verbose,
            service_settings=service_settings,
        )

        if service_settings:
            log_info(message=f"Service setting file [bold magenta]{service_settings.FilePath.name}[/bold magenta] saved")
            service_settings.save()

        return data
