from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.enums.LabelChecker import LabelChecker
from src.logging import log_error, log_info, log_warning
from src.schemas.LabelChecker import LabelCheckerData
from src.services.config import ClassificationConfig, Config
from src.utils.image import build_image_path
from src.utils.ModelLoader import ModelLoader
from src.utils.ServiceSettings import ServiceSettings, subset_settings

from .utils.Dataloader import Dataloader


class ObjectClassification(ModelLoader):
    config: ClassificationConfig = Config.classification
    service_initialized = False

    def initialize_service(self) -> "ObjectClassification":
        def __init__(self) -> None:
            super().__init__()

        self.model_dir = Path(__file__).resolve().parent / "models"
        if not self.is_model_loaded:
            try:
                self.initialize_model(model_dir=self.model_dir)
                self.service_initialized = True if self.is_model_loaded else False
            except Exception as e:
                log_error(message=str(e))
                exit()  # exit the program if the model cannot be loaded
        return self

    def process_data(
        self,
        data: list[LabelCheckerData],
        data_directory: Path,
        verbose: bool,
        service_settings: ServiceSettings | None,
    ) -> Tuple[list[LabelCheckerData], ServiceSettings | None]:
        if not self.config.object_classification_active:
            log_info(
                message=f"[bold magenta]classification[/bold magenta] switched [bold red]off[/bold red].",
                verbose=verbose,
            )
            if service_settings:
                service_settings.remove(self.__class__.__name__)
            return data, service_settings
        log_info(
            message=f"Running [bold magenta]object classification[/bold magenta].",
            verbose=verbose,
        )
        if not self.service_initialized:
            log_info(
                message=f"Initializing [bold magenta]object classification[/bold magenta] model...",
                verbose=verbose,
            )
            self.initialize_service()

        # Check if all features are present in the data and aren't empty
        if self.model_config.Features:
            if not all([f in data[0].__dict__.keys() and data[0].__dict__[f] is not None for f in self.model_config.Features]):
                log_warning(
                    message=f"[bold magenta]Skipping[/bold magenta] classification. {[f for f in self.model_config.Features if f in data[0].__dict__.keys() and data[0].__dict__[f] is None]} features are required in the data for classification with the selected model {self.model_config.Name}."
                )
                return data, service_settings

        data = _classify_data(
            data=data,
            directory=data_directory,
            service_instance=self,
            data_loader=Dataloader(model_config=self.model_config),
            verbose=verbose,
        )

        # update service settings if set for saving
        if service_settings:
            service_settings.update(
                settings={
                    self.__class__.__name__: {
                        "settings": subset_settings(self.config.model_dump(), self.__class__.__name__),
                        "model": self.model_config.model_dump(),
                    },
                },
                preprocess=False,
            )

        return data, service_settings


def _classify_data(
    data: list[LabelCheckerData],
    directory: Path,
    service_instance: ObjectClassification,
    data_loader: Dataloader,
    verbose: bool,
) -> list[LabelCheckerData]:

    # preprocess data
    data_copy = [d.__copy__() for d in data]
    df = _preprocess_data(data=data_copy, directory=directory)
    if df.empty:
        log_error(
            message="No object data to classify found. [bold blue]Please check Preprocessing columns in LabelChecker file[/bold blue] [bold red]Skipping classification[/bold red].",
        )

    # classify data
    ds = data_loader.create_dataloader(df)

    if ds is None:
        log_error(
            message="No data to classify. [bold blue]Please check the Error messages above[/bold blue]. [bold red]Skipping classification[/bold red].",
        )
        return data

    softmax = service_instance.model.predict(ds, verbose=verbose)

    # get class names
    probabilities = np.max(softmax, axis=1)
    predictions = [service_instance.class_names[p + 1] for p in np.argmax(softmax, axis=1)]

    # update data
    for idx, (row_idx, row) in enumerate(df.iterrows()):
        data[row_idx].ProbabilityScore = probabilities[idx]
        data[row_idx].LabelPredicted = predictions[idx]
    return data


# region data preprocessing
def _preprocess_data(data: list[LabelCheckerData], directory: Path) -> pd.DataFrame:
    # build image path
    data = [build_image_path(d, directory) for d in data]

    # create dataframe
    df = pd.DataFrame.from_records([d.to_dict() for d in data])
    # subset object data
    df = _subset_object_data(df=df)

    # drop columns with missing values
    df = df.dropna(axis=1, how="any")
    # Convert columns to float if possible
    df = df.astype(float, errors="ignore")

    return df


def _subset_object_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[
        (df["PreprocessingTrue"].str.lower() == LabelChecker.Preprocessing.OBJECT.value)
        | ((df["PreprocessingTrue"].isnull()) & (df["Preprocessing"].str.lower() == LabelChecker.Preprocessing.OBJECT.value))
    ]
    return df


# endregion
