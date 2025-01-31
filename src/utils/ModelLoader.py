import json
from pathlib import Path

import skops.io as skio
from rich import print
from tensorflow.keras.models import load_model as tf_load_model

from src.logging import log_error, log_info
from src.schemas.ModelConfig import ModelConfig
from src.schemas.ModelInfo import ModelInfo


class ModelLoader:
    def __init__(self):
        super().__init__()
        self.is_model_loaded = False

    def initialize_model(self, model_dir: Path) -> "ModelLoader":

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory [bold red]{model_dir}[/bold red] not found.")
        # get model info and user input
        model_info = self.user_select_model(model_dir=model_dir)

        # load model
        self.model_config = self.load_model_configuration(model_info=model_info)
        self.model = self.load_model(model_info=model_info, model_config=self.model_config)
        self.class_names = self.create_class_names(model_config=self.model_config)
        # set initialized to True
        self.is_model_loaded = True
        return self

    # endregion

    # region user input
    def user_select_model(self, model_dir: Path) -> ModelInfo:
        # get all available models
        available_models = self.get_model_info(model_dir=model_dir)
        if not available_models:
            raise FileNotFoundError("No models found in the model directory.")
        print(available_models)
        if len(available_models) == 1:
            model_info = available_models[1]
            log_info(
                message=f"Automatically selected only model found: [bold blue]{model_info.name}[/bold blue] - version: [bold blue]{model_info.version}[/bold blue]",
                verbose=True,
            )
            return model_info

        # print available models
        log_info(
            message="Available models:",
            verbose=True,
        )
        for idx, model in available_models.items():
            print(f"[bold magenta]{idx}[/bold magenta]: {model.name} - version: {model.version}")

        # get user input
        selected_number = input("Please select model number: ")

        # check if the input is a digit and if the selected number is in the available models
        if not selected_number.isdigit() or int(selected_number) not in available_models:
            log_error(
                message=f"Model number [bold red]{selected_number}[/bold red] not found.",
            )
            return self.user_select_model(model_dir=model_dir)
        model_info = available_models[int(selected_number)]
        log_info(
            message=f"Selected model: [bold blue]{model_info.name}[/bold blue] - version [bold blue]{model_info.version}[/bold blue]",
            verbose=True,
            prefix="Info",
        )
        return model_info

    # endregion

    # region helper methods
    @staticmethod
    def create_class_names(model_config: ModelConfig) -> dict[int, str]:
        return {idx: class_name for idx, class_name in enumerate(model_config.Class_names, start=1)}

    @staticmethod
    def load_model(model_info: ModelInfo, model_config: ModelConfig):
        match model_config.Framework.lower():
            case "tensorflow":
                return ModelLoader.load_tensorflow_model(model_info=model_info)
            case "skops":
                return ModelLoader.load_skops_model(model_info=model_info)

    @staticmethod
    def load_tensorflow_model(model_info: ModelInfo):
        # tf.keras.backend.clear_session()  # clear previous models
        if model_info.file_path.suffix in [".h5", ".hdf5"]:
            model = tf_load_model(model_info.file_path, compile=False)
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            model.load_weights(model_info.file_path.parent / f"model.weights{model_info.file_path.suffix}")
        elif model_info.file_path.suffix in [".keras"]:
            model = tf_load_model(model_info.file_path, compile=True)
        else:
            raise ValueError(f"Model file format [bold red]{model_info.file_path.suffix}[/bold red] not supported.")
        return model

    @staticmethod
    def load_skops_model(model_info: ModelInfo):
        return skio.load(model_info.file_path)

    @staticmethod
    def load_model_configuration(model_info: ModelInfo) -> ModelConfig:
        try:
            with open(model_info.config_file, "r") as file:
                model_config = json.load(file)
                return ModelConfig(model_config=model_config)
        except ValueError as e:
            raise ValueError(f"Model configuration file is missing required values: {e}")

    @staticmethod
    def get_model_info(model_dir: Path) -> dict[int, ModelInfo]:
        model_paths = [model_path for model_path in model_dir.glob("**/model.*") if not model_path.name.startswith("model.weights")]
        if not model_paths:
            raise FileNotFoundError(f"No models found in model directory: [bold red]{model_dir}[/bold red] - exiting.")

        model_paths = ModelLoader.sort_model_paths(model_paths=model_paths)
        return {
            idx: ModelInfo(
                name=model_path.parent.parent.stem,  # models/<model_name>/<model_version>/model.*
                version=model_path.parent.stem,  # models/<model_name>/<model_version>/model.*
                file_path=model_path,
                config_file=model_path.parent.glob("*.json").__next__(),
            )
            for idx, model_path in enumerate(model_paths, start=1)
        }

    @staticmethod
    def sort_model_paths(model_paths: list[Path]) -> list[Path]:
        return sorted(
            model_paths,
            key=lambda x: (
                x.parent.parent.stem,
                x.parent.stem,
            ),
        )

    @staticmethod
    def get_model_path(model_dir: Path, model_name_and_version: dict[str, str]) -> Path:
        model_name = model_name_and_version["name"]
        model_version = model_name_and_version["version"]
        model_paths = model_dir.glob(f"{model_name}/{model_version}/model.*")
        model_path = next(model_paths, None)
        if not model_path:
            raise FileNotFoundError(f"Model [bold red]{model_name}[/bold red] with version [bold red]{model_version}[/bold red] not found.")
        return model_path
