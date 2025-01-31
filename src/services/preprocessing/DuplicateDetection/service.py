import os
from typing import Callable, List, NamedTuple, Optional, Tuple

import pandas as pd
from parallel_pandas import ParallelPandas

from src.enums.LabelChecker import LabelChecker
from src.logging import log_error, log_info, log_warning
from src.schemas.LabelChecker import LabelCheckerData
from src.services.config import Config, PreprocessingConfig
from src.utils.ServiceSettings import ServiceSettings, subset_settings

from .utils.bbox_overlap import BboxOverlap
from .utils.get_metric import get_metric

# initialize parallel-pandas
ParallelPandas.initialize(n_cpu=os.cpu_count(), split_factor=4, disable_pr_bar=True)


class DuplicateDetection:
    config: PreprocessingConfig = Config.preprocessing

    @classmethod
    def process_data(
        cls,
        data: List[LabelCheckerData],
        verbose: bool,
        service_settings: ServiceSettings | None,
    ) -> Tuple[list[LabelCheckerData], ServiceSettings | None]:
        if not cls.config.duplicate_detection_active:
            log_info(
                message=f"[bold magenta]duplicate detection[/bold magenta] switched [bold red]off[/bold red].",
                verbose=verbose,
            )
            if service_settings:
                service_settings.remove(cls.__name__)
            return data, service_settings

        log_info(
            message=f"""Running [bold magenta]duplicate detection[/bold magenta] with settings:
                - [bold magenta]max_source_distance[/bold magenta]: {cls.config.duplicate_detection_max_source_distance}
                - [bold magenta]x_coord_offset[/bold magenta]: {cls.config.duplicate_detection_x_coordinate_offset}
                - [bold magenta]overlap_threshold[/bold magenta]: {cls.config.duplicate_detection_overlap_threshold}
                - [bold magenta]metric[/bold magenta]: {cls.config.duplicate_detection_metric}
                - [bold magenta]metric_threshold[/bold magenta]: {cls.config.duplicate_detection_metric_threshold}
                - [bold magenta]similarity_features[/bold magenta]: {cls.config.duplicate_detection_similarity_features}

        objects who's bounding box overlap [bold magenta]{cls.config.duplicate_detection_overlap_threshold}[/bold magenta] or more, or with a [bold magenta]{cls.config.duplicate_detection_metric}[/bold magenta] above [bold magenta]{cls.config.duplicate_detection_metric_threshold}[/bold magenta] will be marked as duplicates [bold magenta]{LabelChecker.Preprocessing.DUPLICATE.value}[/bold magenta].
                """,
            verbose=verbose,
        )

        # update service settings if set for saving
        if service_settings:
            service_settings.update({cls.__name__: subset_settings(cls.config.model_dump(), cls.__name__)})

        df = pd.DataFrame.from_records([lc_data.to_dict() for lc_data in data]).convert_dtypes()
        df = _detect_duplicates(df=df, config=cls.config)
        return [LabelCheckerData(series=series) for _, series in df.iterrows()], service_settings


def _detect_duplicates(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    # Initialize parameters
    max_source_distance = config.duplicate_detection_max_source_distance
    x_coord_offset = config.duplicate_detection_x_coordinate_offset
    overlap_threshold = config.duplicate_detection_overlap_threshold
    metric = get_metric(config=config)
    metric_threshold = config.duplicate_detection_metric_threshold
    similarity_features = config.duplicate_detection_similarity_features

    # duplicate label
    duplicate_label = LabelChecker.Preprocessing.DUPLICATE.value

    # Define the labels to process
    labels_to_process = [
        LabelChecker.Preprocessing.OBJECT.value,
        duplicate_label,
    ]

    # Check if the similarity features are in the dataframe
    similarity_features = list(set(similarity_features).intersection(df.columns))
    if not similarity_features:
        log_error(
            message=f"No similarity features values found in the dataframe, [bold magenta]Skipping[/bold magenta] similarity check",
        )
    if df[similarity_features].isnull().values.any():
        log_warning(
            message=f"[bold magenta]Missing[/bold magenta] similarity feature values: {[feature for feature in similarity_features if df[feature].isnull().any()]}",
        )
        log_info(
            message=f"Making similarity check with remaining features: {[feature for feature in similarity_features if not df[feature].isnull().any()]}",
            verbose=True,
        )
        similarity_features = [feature for feature in similarity_features if not df[feature].isnull().any()]

    # Process the rows in parallel
    duplicate_indices = df.p_apply(
        _process_row,
        axis=1,
        args=(
            df,
            labels_to_process,
            max_source_distance,
            x_coord_offset,
            overlap_threshold,
            metric,
            metric_threshold,
            similarity_features,
        ),
    )
    for row in duplicate_indices:
        if row:
            for idx in row:
                if not df.at[idx, "Preprocessing"] in [
                    LabelChecker.Preprocessing.DUPLICATE.value,
                    LabelChecker.Preprocessing.OBJECT.value,
                ]:
                    continue
                df.at[idx, "Preprocessing"] = duplicate_label
    return df


def _process_row(
    row,
    df: pd.DataFrame,
    labels_to_process: List[str],
    max_source_distance: int,
    x_coord_offset: int,
    overlap_threshold: float,
    metric: Callable,
    metric_threshold: float,
    similarity_features: List[str],
) -> Optional[list[int]]:
    # Define the labels to process
    if row.Preprocessing not in labels_to_process:
        return None
    # Get the rows within the distance and offset
    selected_rows = _get_rows_within_distance_and_offset(
        df=df,
        source_image_nr=row.SrcImage,
        source_x_coord=row.SrcX,
        max_source_distance=max_source_distance,
        x_coord_offset=x_coord_offset,
    )
    # check if rows after filtering are empty
    if selected_rows.empty:
        return None

    # Check if the any of the rows are a duplicate
    missing_feature_warning = True
    duplicate_indices = []
    for selected_row in selected_rows.itertuples():
        # bounding box overlap check
        if _is_overlapping_duplicate(
            row=row,
            selected_row=selected_row,
            threshold=overlap_threshold,
        ):
            duplicate_indices.append(selected_row.Index)
            continue

        # similarity check
        if not similarity_features:
            continue

        row_features = row[similarity_features]
        selected_row_features = df.loc[selected_row.Index, similarity_features]
        if (
            _calculate_similarity(
                row_features=row_features,
                selected_row_features=selected_row_features,
                metric=metric,
            )
            >= metric_threshold
        ):
            duplicate_indices.append(selected_row.Index)
    return duplicate_indices


def _get_rows_within_distance_and_offset(
    df: pd.DataFrame,
    source_image_nr: int,
    source_x_coord: int,
    max_source_distance: int,
    x_coord_offset: int,
) -> pd.DataFrame:
    return df[
        df["SrcImage"].between(source_image_nr + 1, source_image_nr + max_source_distance)
        & df["SrcX"].between(source_x_coord - x_coord_offset, source_x_coord + x_coord_offset)
    ]


def _is_overlapping_duplicate(
    row: pd.Series,
    selected_row: NamedTuple,
    threshold: float,
) -> bool:
    return BboxOverlap.bbox_overlap_check(row1=row, row2=selected_row, iou_threshold=threshold)


def _calculate_similarity(
    row_features,
    selected_row_features: int,
    metric: Callable,
) -> bool:
    return metric(
        row_features,
        selected_row_features,
    )
