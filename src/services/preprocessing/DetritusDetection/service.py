from pathlib import Path
from typing import Tuple

import pandas as pd

from src.enums.LabelChecker import LabelChecker
from src.logging import log_error, log_info, log_warning
from src.schemas.LabelChecker import LabelCheckerData
from src.services.config import Config, PreprocessingConfig
from src.utils.ModelLoader import ModelLoader
from src.utils.ServiceSettings import ServiceSettings, subset_settings


class DetritusDetection(ModelLoader):
    config: PreprocessingConfig = Config.preprocessing
    service_initialized = False

    def initialize_service(self) -> "DetritusDetection":
        def __init__(self) -> None:
            super().__init__()

        self.model_dir = Path(__file__).resolve().parent / "models"
        if not self.is_model_loaded:
            try:
                self.initialize_model(model_dir=self.model_dir)
                self.service_initialized = True if self.is_model_loaded else False
            except Exception as e:
                log_error(message=str(e))
        return self

    def process_data(
        self,
        data: list[LabelCheckerData],
        verbose: bool,
        service_settings: ServiceSettings | None,
    ) -> Tuple[list[LabelCheckerData], ServiceSettings | None]:
        if not self.config.detritus_detection_active:
            log_info(
                message=f"[bold magenta]detritus detection[/bold magenta] switched [bold red]off[/bold red].",
                verbose=verbose,
            )
            if service_settings:
                service_settings.remove(self.__class__.__name__)
            return data, service_settings
        log_info(
            message=f"""Running [bold magenta]detritus detection[/bold magenta].""",
            verbose=verbose,
        )
        if not self.service_initialized:
            log_info(
                message=f"Initializing [bold magenta]detritus detection[/bold magenta] model...",
                verbose=verbose,
            )
            self.initialize_service()

        # Check if all features are present in the data and aren't empty
        if self.model_config.Features:
            if not all([f in data[0].__dict__.keys() and data[0].__dict__[f] is not None for f in self.model_config.Features]):
                log_warning(
                    message=f"[bold magenta]Skipping[/bold magenta] detritus detection. {[f for f in self.model_config.Features if f in data[0].__dict__.keys() and data[0].__dict__[f] is None]} features are required in the data for detritus detection."
                )
                return data, service_settings

        data = _detect_detritus(
            data=data,
            service_instance=self,
            verbose=verbose,
        )

        # update service settings if set for saving
        if service_settings:
            service_settings.update(
                {
                    self.__class__.__name__: {
                        "settings": subset_settings(self.config.model_dump(), self.__class__.__name__),
                        "model": self.model_config.model_dump(),
                    },
                }
            )
        return data, service_settings


def _detect_detritus(
    data: list[LabelCheckerData],
    service_instance: DetritusDetection,
    verbose: bool,
) -> list[LabelCheckerData]:
    # preprocess data
    data_copy = [d.__copy__() for d in data]
    df = _preprocess_data(data=data_copy)

    # get predictions
    predictions = service_instance.model.predict(df[service_instance.model_config.Features])
    # get class names from binary predictions
    predictions = [(service_instance.class_names.get(2) if p == 1 else service_instance.class_names.get(1)) for p in predictions]  # 2 is object, 1 is bubble

    # update data
    for idx, (row_idx, row) in enumerate(df.iterrows()):
        data[row_idx].Preprocessing = predictions[idx]
    return data


# region data preprocessing
def _preprocess_data(data: list[LabelCheckerData]) -> pd.DataFrame:
    # create dataframe
    df = pd.DataFrame.from_records([d.to_dict() for d in data])

    # subset object data
    df = _subset_object_data(df=df)

    # drop any column with a missing values
    df = df.dropna(axis=1, how="any")

    # Convert columns to float if possible
    for column in df.columns:
        try:
            df[column] = df[column].astype(float)
        except ValueError:
            pass
    return df


def _subset_object_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[
        (
            df["Preprocessing"].isin(
                [
                    LabelChecker.Preprocessing.OBJECT.value,
                    LabelChecker.Preprocessing.BUBBLE.value,  # for reprocessing
                    LabelChecker.Preprocessing.DUPLICATE.value,  # for reprocessing
                ]
            )
        )
    ]
    return df


# endregion
