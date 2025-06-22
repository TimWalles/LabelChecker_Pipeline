import re
import uuid
from typing import Tuple

import pandas as pd

from ..structure.flowcam_data_structure import flocam_data_structure


def normalize_flowcam_data(
    data: pd.DataFrame,
    sample_name: str,
    reprocess: bool,
    running_classification: bool,
) -> pd.DataFrame:
    if reprocess and not running_classification:
        # if data is already normalized, reset the preprocessing column
        data["Preprocessing"] = ["object" for _ in range(len(data))]
    data.columns = [normalize_column_names(name) for name in data.columns]
    missing_cols = check_columns(data)
    if missing_cols:
        data = add_missing_columns(data, missing_cols=missing_cols)
    data = check_name(data=data, sample_name=sample_name)
    data = check_image_file(data=data, sample_name=sample_name)
    data = check_uuid(data=data)
    data = check_source_image(data=data)
    return data


# region normalize column names
def normalize_column_names(name: str) -> str:
    # name = name.replace("/", "")
    name = re.sub(r"[\{}()/.]", "", name)
    if " " in name or name.isupper():
        name = camel_case(name)
    name = normalize_column_name(name)
    return name


def camel_case(s):
    return "".join(t.title() for t in s.split())


def normalize_column_name(name: str) -> str:
    match name:
        case "AreaAbd":
            new_name = "AbdArea"
        case "AreaFilled":
            new_name = "FilledArea"
        case "AverageBlue":
            new_name = "AvgBlue"
        case "AverageGreen":
            new_name = "AvgGreen"
        case "AverageRed":
            new_name = "AvgRed"
        case "CaptureId" | "ParticleId":
            new_name = "Id"
        case "CaptureX":
            new_name = "SrcX"
        case "CaptureY":
            new_name = "SrcY"
        case "CalibrationFactor":
            new_name = "CalConst"
        case "CalibrationImage":
            new_name = "CalImage"
        case "Ch1PeakArea":
            new_name = "Ch1Area"
        case "Ch2PeakArea":
            new_name = "Ch2Area"
        case "Ch3PeakArea":
            new_name = "Ch3Area"
        case "DiameterAbd":
            new_name = "AbdDiameter"
        case "DiameterEsd":
            new_name = "EsdDiameter"
        case "DiameterFd":
            new_name = "FdDiameter"
        case "FeretAngleMax":
            new_name = "FeretMaxAngle"
        case "FeretAngleMin":
            new_name = "FeretMinAngle"
        case "Filename" | "ImageFile":
            new_name = "CollageFile"
        case "OriginalReferenceId":
            new_name = "Uuid"
        case "ParticlesPerChain":
            new_name = "Ppc"
        case "PixelW" | "ImageWidth":
            new_name = "ImageW"
        case "PixelH" | "ImageHeight":
            new_name = "ImageH"
        case "SaveX":
            new_name = "ImageX"
        case "SaveY":
            new_name = "ImageY"
        case "SourceImage":
            new_name = "SrcImage"
        case "VolumeAbd":
            new_name = "AbdVolume"
        case "VolumeEsd":
            new_name = "EsdVolume"
        case _:
            new_name = name
    return new_name


def check_columns(df: pd.DataFrame) -> list:
    return [name for name in flocam_data_structure.keys() if name not in df.columns]


def add_missing_columns(
    df: pd.DataFrame,
    missing_cols: list,
) -> pd.DataFrame:
    new_columns = [*df.columns, *missing_cols]
    return df.reindex(columns=new_columns).convert_dtypes(dtype_backend="pyarrow")


# endregion

# region check and assign missing data


def check_name(data: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    if all(data["Name"].isnull()):
        data["Name"] = data["Name"].map(
            lambda name: _assign_name(
                name=name,
                sample_name=sample_name,
            )
        )
    return data


def _assign_name(
    name: str,
    sample_name: str,
) -> str:
    return sample_name if pd.isna(name) else name


def check_uuid(data: pd.DataFrame) -> pd.DataFrame:
    if all(data["Uuid"].isnull()):
        data["Uuid"] = data["Uuid"].map(lambda uid: _assign_uuid(uid=uid))
    return data


def _assign_uuid(uid: str) -> uuid.UUID:
    if pd.isna(uid):
        return uuid.uuid1()
    try:
        return uuid.UUID(uid)
    except TypeError:
        return uuid.uuid1()


def check_image_file(data: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    data["ImageFilename"] = data.apply(
        lambda df: _assign_filename(
            row=df,
            sample_name=sample_name,
            n_rows=data.shape[0],
        ),
        axis=1,
    )
    return data


def _assign_filename(
    row: pd.Series,
    sample_name: str,
    n_rows: int,
) -> pd.Series:
    return pd.Series(
        [
            (
                f"{sample_name}_{str(row['Id']-1).zfill(len(str(n_rows)) + 1)}.png"  # <=9999 n_rows -> 5; digits <=99999 -> 6 digits; ...
                if pd.isnull(row["ImageFilename"]) and pd.isnull(row["CollageFile"])
                else (row["ImageFilename"] if not pd.isnull(row["ImageFilename"]) else None)
            )
        ]
    )


def check_source_image(data: pd.DataFrame) -> pd.DataFrame:
    if all(data["SrcImage"].isnull()):
        source_y_old = None
        source_image_nr = 1
        SrcImage = []
        for src_y in data["SrcY"]:
            source_y_old, source_image_nr = assign_source_image(
                SrcY=src_y,
                source_y_old=source_y_old,
                source_image_nr=source_image_nr,
            )
            SrcImage.append(source_image_nr)
        data["SrcImage"] = SrcImage
    return data


def assign_source_image(
    SrcY: int,
    source_y_old: int | None,
    source_image_nr: int,
) -> Tuple[int, int]:
    if not source_y_old:
        return (1, 1)
    if SrcY > source_y_old:
        source_image_nr += 1
    return (
        SrcY,
        source_image_nr,
    )


# endregion
