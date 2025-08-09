import os
from pathlib import Path

import pandas as pd
import typer
from rich.progress import track
from typing_extensions import Annotated

from src.enums.Instruments import Instruments
from src.error.handler import Tracker, error_handler
from src.instruments.list_data import fetch_data_files
from src.instruments.read_data import read_data
from src.logging import log_error, log_info, log_progress
from src.services.ProcessData import DataClassifier
from src.utils.dataframe import order_df_columns


def main(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data_dir",
            "-D",
            exists=True,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
            help="Path to data directory",
        ),
    ],
    instrument: Annotated[
        Instruments,
        typer.Option(
            "--instrument",
            "-I",
            case_sensitive=False,
            help="please select the instrument",
        ),
    ] = Instruments.FLOWCAM,
    save_settings: Annotated[
        bool,
        typer.Option(
            "--save_settings",
            help="Save services settings inside each processed sample folder",
        ),
    ] = True,
    reprocess: Annotated[
        bool,
        typer.Option("--reprocess", "-R", help="set to reprocess already processed data"),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-V")] = False,
):
    log_progress(message=f"Starting [bold magenta]{instrument}[/bold magenta] data-Classification pipeline")
    # region informing user about the services
    if reprocess and verbose:
        log_info(
            message="script set to reprocess existing labelchecker files",
        )

    if save_settings:
        log_info(
            message="The settings of used services will be [bold magenta]saved[/bold magenta]",
            verbose=verbose,
        )
    # endregion

    # region fetch data files
    sample_files = fetch_data_files(
        path=data_dir,
        instrument=instrument,
        reprocess=reprocess,
        running_classification=True,
    )
    if not sample_files:
        log_error(
            message="No samples were found to process. Exiting the script.",
        )
        exit()
    log_info(
        message=f"found {len(sample_files)} samples to process",
    )
    # endregion

    # region processing data
    tracker = Tracker()
    data_classifier = DataClassifier()
    for file_path in track(sample_files.values(), description="[bold green][Progress update][/bold green]"):
        log_progress(message=f"processing sample [bold magenta]{file_path.name}[/bold magenta]")

        # region read data
        try:
            data, tracker = read_data(
                file_path=file_path,
                instrument=instrument,
                running_classification=True,
                tracker=tracker,
            )
        except Exception as e:
            log_error(message=f"Failed to read data from {file_path.name}: {str(e)}")
            tracker = error_handler(
                tracker=tracker,
                name=file_path.name,
                desc=str(e),
            )
            continue
        if not data:
            continue
        # endregion

        # region classification services
        try:
            data = data_classifier.classify_label_checker_data(
                data=data,
                data_directory=file_path.parent,
                verbose=verbose,
                save_settings=save_settings,
            )
        except Exception as e:
            log_error(message=f"Failed to classify data from {file_path.name}: {str(e)}. Continuing to the next file")
            tracker = error_handler(
                tracker=tracker,
                name=file_path.name,
                desc=str(e),
            )
            continue
            # endregion

        # region write data
        try:
            data_df = pd.DataFrame.from_records([lc_data.to_dict() for lc_data in data])
            data_df = order_df_columns(df=data_df)
            data_df.to_csv(
                os.path.join(
                    file_path.parent.as_posix(),
                    "LabelChecker_" + file_path.parent.name + ".csv",
                ),
                index=False,
            )
        except Exception as e:
            log_error(message=f"Failed to write data to {"LabelChecker_" + file_path.parent.name + ".csv"}: {str(e)}. Continuing to the next file")
            tracker = error_handler(
                tracker=tracker,
                name=file_path.name,
                desc=str(e),
            )
            continue
        # endregion
    # endregion

    # region reporting results
    log_progress(message=f"processed [bold magenta]{tracker.successful}[/bold magenta] nr. of samples successful")

    if tracker.failed:
        log_error(message=f"failed processing [bold magenta]{len(tracker.failed)}[/bold magenta] nr. of samples")
        log_error(
            message=f"Failed samples: {', '.join([f.name for f in tracker.failed])}",
        )
    log_progress(message=f"Finished [bold magenta]{instrument}[/bold magenta] data-Classification pipeline. Exiting the script.")
    # endregion


if __name__ == "__main__":
    typer.run(main)
