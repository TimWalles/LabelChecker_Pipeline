import pandas as pd


def order_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "Name",
            "Date",
            "Time",
            "CollageFile",
            "ImageFilename",
            "Id",
            "GroupId",
            "Uuid",
            "SrcImage",
            "SrcX",
            "SrcY",
            "ImageX",
            "ImageY",
            "ImageW",
            "ImageH",
            "Timestamp",
            "ElapsedTime",
            "CalConst",
            "CalImage",
            "AbdArea",
            "AbdDiameter",
            "AbdVolume",
            "AspectRatio",
            "AvgBlue",
            "AvgGreen",
            "AvgRed",
            "BiovolumeCylinder",
            "BiovolumePSpheroid",
            "BiovolumeSphere",
            "Ch1Area",
            "Ch1Peak",
            "Ch1Width",
            "Ch2Area",
            "Ch2Ch1Ratio",
            "Ch2Peak",
            "Ch2Width",
            "Ch3Area",
            "Ch3Peak",
            "Ch3Width",
            "CircleFit",
            "Circularity",
            "CircularityHu",
            "Compactness",
            "ConvexPerimeter",
            "Convexity",
            "EdgeGradient",
            "Elongation",
            "EsdDiameter",
            "EsdVolume",
            "FdDiameter",
            "FeretMaxAngle",
            "FeretMinAngle",
            "FiberCurl",
            "FiberStraightness",
            "FilledArea",
            "FilterScore",
            "GeodesicAspectRatio",
            "GeodesicLength",
            "GeodesicThickness",
            "Intensity",
            "Length",
            "Perimeter",
            "Ppc",
            "RatioBlueGreen",
            "RatioRedBlue",
            "RatioRedGreen",
            "Roughness",
            "ScatterArea",
            "ScatterPeak",
            "SigmaIntensity",
            "SphereComplement",
            "SphereCount",
            "SphereUnknown",
            "SphereVolume",
            "SumIntensity",
            "Symmetry",
            "Transparency",
            "Width",
            "BiovolumeMS",
            "SurfaceAreaMS",
            "Preprocessing",
            "PreprocessingTrue",
            "LabelPredicted",
            "ProbabilityScore",
            "LabelTrue",
        ]
    ]
