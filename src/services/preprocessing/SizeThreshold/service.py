from typing import Tuple

from src.enums.LabelChecker import LabelChecker
from src.logging import log_error, log_info
from src.schemas.LabelChecker import LabelCheckerData
from src.services.config import Config
from src.utils.ServiceSettings import ServiceSettings, subset_settings


class SizeThreshold:
    config = Config.preprocessing

    @classmethod
    def process_data(
        cls,
        data: list[LabelCheckerData],
        verbose: bool,
        service_settings: ServiceSettings | None,
    ) -> Tuple[list[LabelCheckerData], ServiceSettings | None]:
        if not cls.config.size_threshold_active:
            log_info(
                message=f"[bold magenta]size threshold[/bold magenta] switched [bold red]off[/bold red].",
                verbose=verbose,
            )
            if service_settings:
                service_settings.remove(cls.__name__)
            return (data, service_settings)
        if not all([(True if lc_data.get_value(cls.config.size_threshold_threshold_variable) != None else False) for lc_data in data]):
            log_error(
                message=f"""Running [bold magenta]size threshold[/bold magenta] but data does not contain [bold magenta]{cls.config.size_threshold_threshold_variable}[/bold magenta]
        Skipping size detection.
                """,
                verbose=verbose,
            )
            return data, service_settings

        log_info(
            message=f"""Running [bold magenta]size threshold[/bold magenta] with settings:
        threshold variable: [bold magenta]{cls.config.size_threshold_threshold_variable}[/bold magenta]
        lower threshold value: [bold magenta]{cls.config.size_threshold_lower_bound}[/bold magenta] set to [bold magenta]{LabelChecker.Preprocessing.SMALL.value}[/bold magenta]
        upper threshold value: [bold magenta]{cls.config.size_threshold_upper_bound}[/bold magenta] set to [bold magenta]{LabelChecker.Preprocessing.LARGE.value}[/bold magenta]""",
            verbose=verbose,
        )

        # update service settings if set for saving
        if service_settings:
            service_settings.update({cls.__name__: subset_settings(cls.config.model_dump(), cls.__name__)})

        return [
            _check_conditions(
                lc_data=lc_data,
                threshold_variable=cls.config.size_threshold_threshold_variable,
                lower_bound=cls.config.size_threshold_lower_bound,
                upper_bound=cls.config.size_threshold_upper_bound,
            )
            for lc_data in data
        ], service_settings


def _check_conditions(
    lc_data: LabelCheckerData,
    threshold_variable: str,
    lower_bound: float,
    upper_bound: float,
) -> LabelCheckerData:
    if not lc_data.Preprocessing or lc_data.Preprocessing in [LabelChecker.Preprocessing.OBJECT.value]:
        threshold_value = lc_data.get_value(threshold_variable)
        if threshold_value and threshold_value < lower_bound:
            lc_data.Preprocessing = LabelChecker.Preprocessing.SMALL.value
            lc_data.PreprocessingTrue = LabelChecker.Preprocessing.SMALL.value
            return lc_data

        if threshold_value and threshold_value > upper_bound:
            lc_data.Preprocessing = LabelChecker.Preprocessing.LARGE.value
            lc_data.PreprocessingTrue = LabelChecker.Preprocessing.LARGE.value
            return lc_data

    lc_data.Preprocessing = lc_data.Preprocessing if lc_data.Preprocessing else LabelChecker.Preprocessing.OBJECT.value
    lc_data.PreprocessingTrue = (
        ""
        if lc_data.PreprocessingTrue
        in [
            LabelChecker.Preprocessing.SMALL.value,
            LabelChecker.Preprocessing.LARGE.value,
        ]
        else lc_data.PreprocessingTrue
    )
    return lc_data
