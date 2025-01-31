import json
from pathlib import Path
from typing import Protocol, Union


def subset_settings(settings: dict, name: str) -> dict:
    name = camel_case_to_snake_case(name)
    return {key: settings[key] for key in settings if key.startswith(name)}


def camel_case_to_snake_case(name: str) -> str:
    return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")


class ServiceSettings:
    Settings: dict = {
        "Preprocessing": {},
        "Classification": {},
    }
    FilePath: Path

    def __init__(self, directory: Path) -> None:
        self.Settings = self.get(directory=directory)

    def get(self, directory: Path):
        self.FilePath = directory.joinpath("processing_settings_summary.txt")
        if self.FilePath.exists():
            return self.read()
        return self.Settings

    def read(self) -> dict:
        with open(self.FilePath, "r") as file:
            settings = json.loads(file.read())
        return settings

    def update(self, settings: dict, preprocess: bool = True) -> None:
        if settings:
            if preprocess:
                self.Settings["Preprocessing"].update(settings)
            else:
                self.Settings["Classification"].update(settings)

    def remove(self, key: str, preprocess: bool = True) -> None:
        if preprocess:
            if key in self.Settings["Preprocessing"]:
                self.Settings["Preprocessing"].__delitem__(key)
        else:
            if key in self.Settings["Classification"]:
                self.Settings["Classification"].__delitem__(key)

    def save(self):
        with open(self.FilePath, "w") as file:
            json.dump(self.Settings, file, indent=4)
