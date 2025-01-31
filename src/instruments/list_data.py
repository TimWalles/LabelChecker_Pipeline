from pathlib import Path
from typing import List, Optional, Dict
from rich import print
from src.enums.Instruments import Instruments


def fetch_data_files(
    path: Path,
    instrument: Instruments,
    reprocess: bool,
    running_classification: bool = False,
) -> Dict[str, Path]:
    match instrument:
        case Instruments.FLOWCAM.value:
            try:
                directory = Path(path)
                if not directory.is_dir():
                    raise ValueError(f"'{path}' is not a valid directory path.")
                files = {}

                unique_directories = set()
                for file_path in list(directory.glob(f"**/*.csv")):
                    unique_directories.add(file_path.parent)

                for directory_path in unique_directories:
                    if directory_path.name in files:
                        continue

                    processed = is_processed(directory_path)
                    if processed and not reprocess and not running_classification:
                        continue

                    if processed and reprocess or processed and running_classification:
                        file = is_labelchecker_csv(directory_path=directory_path)
                        if file:
                            files[directory_path.name] = file
                            continue
                    file = is_data_csv(directory_path=directory_path)
                    if file:
                        files[directory_path.name] = file
                return files
            except Exception as e:
                print(f"[bold red][Warning][/bold red]: {e}. Exiting the script.")
                exit()
        case _:
            return {}


def is_processed(directory: Path) -> bool:
    return (
        True
        if any(
            [
                file_path.is_file() and file_path.name.startswith("LabelChecker_")
                for file_path in directory.iterdir()
            ]
        )
        else False
    )


def is_labelchecker_csv(directory_path: Path) -> Optional[Path]:
    return next(
        (f for f in directory_path.glob("LabelChecker_*.csv") if f.is_file()), None
    )


# region Flowcam
def is_data_csv(directory_path: Path) -> Optional[Path]:
    return next(
        (f for f in directory_path.glob(f"{directory_path.name}.csv") if f.is_file()),
        None,
    )


# endregion
