from datetime import date, datetime, time
from types import NoneType
from typing import Any, Optional
from uuid import UUID

import pandas as pd


class LabelCheckerData:
    def __init__(self, series: pd.Series) -> None:
        def get_series_value(series, column, default=None):
            # Check if column exists and value is not NaN, otherwise return default
            return (
                series[column]
                if column in series and not pd.isna(series[column])
                else default
            )

        # region meta data
        self.Name: str | None = get_series_value(series, "Name")
        self.Date: date | None = get_series_value(series, "Date")
        self.Time: time | None = get_series_value(series, "Time")
        self.CollageFile: str | None = get_series_value(series, "CollageFile")
        self.ImageFilename: str | None = get_series_value(series, "ImageFilename")
        self.Id: int = series["Id"]
        self.GroupId: int | None = get_series_value(series, "GroupId")
        self.Uuid: UUID | None = get_series_value(series, "Uuid")
        self.SrcImage: int = series["SrcImage"]
        self.SrcX: int = series["SrcX"]
        self.SrcY: int = series["SrcY"]
        self.ImageX: int | None = get_series_value(series, "ImageX")
        self.ImageY: int | None = get_series_value(series, "ImageY")
        self.ImageW: int = series["ImageW"]
        self.ImageH: int = series["ImageH"]
        self.Timestamp: datetime | None = get_series_value(series, "Timestamp")
        self.ElapsedTime: datetime | None = get_series_value(series, "ElapsedTime")
        self.CalConst: float = get_series_value(series, "CalConst", 0.0)
        self.CalImage: int = get_series_value(series, "CalImage", 0)
        # endregion

        # region object data
        self.AbdArea: float | None = get_series_value(series, "AbdArea")
        self.AbdDiameter: float | None = get_series_value(series, "AbdDiameter")
        self.AbdVolume: float | None = get_series_value(series, "AbdVolume")
        self.AspectRatio: float | None = get_series_value(series, "AspectRatio")
        self.AvgBlue: float | None = get_series_value(series, "AvgBlue")
        self.AvgGreen: float | None = get_series_value(series, "AvgGreen")
        self.AvgRed: float | None = get_series_value(series, "AvgRed")
        self.BiovolumeCylinder: float | None = get_series_value(
            series, "BiovolumeCylinder"
        )
        self.BiovolumePSpheroid: float | None = get_series_value(
            series, "BiovolumePSpheroid"
        )
        self.BiovolumeSphere: float | None = get_series_value(series, "BiovolumeSphere")
        self.Ch1Area: float | None = get_series_value(series, "Ch1Area")
        self.Ch1Peak: float | None = get_series_value(series, "Ch1Peak")
        self.Ch1Width: float | None = get_series_value(series, "Ch1Width")
        self.Ch2Area: float | None = get_series_value(series, "Ch2Area")
        self.Ch2Peak: float | None = get_series_value(series, "Ch2Peak")
        self.Ch2Width: float | None = get_series_value(series, "Ch2Width")
        self.Ch3Area: float | None = get_series_value(series, "Ch3Area")
        self.Ch3Peak: float | None = get_series_value(series, "Ch3Peak")
        self.Ch3Width: float | None = get_series_value(series, "Ch3Width")
        self.Ch2Ch1Ratio: float | None = get_series_value(series, "Ch2Ch1Ratio")
        self.CircleFit: float | None = get_series_value(series, "CircleFit")
        self.Circularity: float | None = get_series_value(series, "Circularity")
        self.CircularityHu: float | None = get_series_value(series, "CircularityHu")
        self.Compactness: float | None = get_series_value(series, "Compactness")
        self.Convexity: float | None = get_series_value(series, "Convexity")
        self.ConvexPerimeter: float | None = get_series_value(series, "ConvexPerimeter")
        self.EdgeGradient: float | None = get_series_value(series, "EdgeGradient")
        self.Elongation: float | None = get_series_value(series, "Elongation")
        self.EsdDiameter: float | None = get_series_value(series, "EsdDiameter")
        self.EsdVolume: float | None = get_series_value(series, "EsdVolume")
        self.FdDiameter: float | None = get_series_value(series, "FdDiameter")
        self.FeretMaxAngle: float | None = get_series_value(series, "FeretMaxAngle")
        self.FeretMinAngle: float | None = get_series_value(series, "FeretMinAngle")
        self.FiberCurl: float | None = get_series_value(series, "FiberCurl")
        self.FiberStraightness: float | None = get_series_value(
            series, "FiberStraightness"
        )
        self.FilledArea: float | None = get_series_value(series, "FilledArea")
        self.FilterScore: float | None = get_series_value(series, "FilterScore")
        self.GeodesicAspectRatio: float | None = get_series_value(
            series, "GeodesicAspectRatio"
        )
        self.GeodesicLength: float | None = get_series_value(series, "GeodesicLength")
        self.GeodesicThickness: float | None = get_series_value(
            series, "GeodesicThickness"
        )
        self.Intensity: float | None = get_series_value(series, "Intensity")
        self.Length: float | None = get_series_value(series, "Length")
        self.Ppc: float | None = get_series_value(series, "Ppc")
        self.Perimeter: float | None = get_series_value(series, "Perimeter")
        self.RatioBlueGreen: float | None = get_series_value(series, "RatioBlueGreen")
        self.RatioRedBlue: float | None = get_series_value(series, "RatioRedBlue")
        self.RatioRedGreen: float | None = get_series_value(series, "RatioRedGreen")
        self.Roughness: float | None = get_series_value(series, "Roughness")
        self.ScatterArea: float | None = get_series_value(series, "ScatterArea")
        self.ScatterPeak: float | None = get_series_value(series, "ScatterPeak")
        self.SigmaIntensity: float | None = get_series_value(series, "SigmaIntensity")
        self.SphereComplement: float | None = get_series_value(
            series, "SphereComplement"
        )
        self.SphereCount: float | None = get_series_value(series, "SphereCount")
        self.SphereUnknown: float | None = get_series_value(series, "SphereUnknown")
        self.SphereVolume: float | None = get_series_value(series, "SphereVolume")
        self.SumIntensity: int | None = get_series_value(series, "SumIntensity")
        self.Symmetry: float | None = get_series_value(series, "Symmetry")
        self.Transparency: float | None = get_series_value(series, "Transparency")
        self.Width: float | None = get_series_value(series, "Width")
        self.BiovolumeMS: float | None = get_series_value(series, "BiovolumeMS")
        self.SurfaceAreaMS: float | None = get_series_value(series, "SurfaceAreaMS")
        self.Preprocessing: str | None = get_series_value(
            series, "Preprocessing", "object"
        )
        self.PreprocessingTrue: str | None = get_series_value(
            series, "PreprocessingTrue"
        )
        self.LabelPredicted: str | None = get_series_value(series, "LabelPredicted")
        self.ProbabilityScore: float | None = get_series_value(
            series, "ProbabilityScore"
        )
        self.LabelTrue: str | None = get_series_value(series, "LabelTrue")
        # endregion

    def get_value(self, attribute_name: str) -> Optional[Any]:
        return getattr(self, attribute_name, None)

    def to_dict(self) -> dict:
        return {
            attr_name: self._normalize_value(getattr(self, attr_name))
            for attr_name in dir(self)
            if not callable(getattr(self, attr_name)) and not attr_name.startswith("__")
        }

    def _normalize_value(self, value: Any):
        if isinstance(value, (str, float, int, NoneType)):
            return value
        if isinstance(value, pd.Timestamp):
            return str(value)
        if isinstance(value, time):
            return f"{value:%H:%M:%S}"
        if isinstance(value, date):
            return f"{value:%Y-%m-%d}"
        if isinstance(value, UUID):
            return str(value)
        return value

    def __repr__(self) -> str:
        return str(self.to_dict())

    def __copy__(self):
        return LabelCheckerData(self.to_dict())
