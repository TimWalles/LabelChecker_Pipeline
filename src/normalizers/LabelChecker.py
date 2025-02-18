import uuid

import pandas as pd

from src.schemas.LabelChecker import LabelCheckerData
from src.schemas.LabelCheckerColumns import column_name_types


class LabelCheckerNormalizer:
    @classmethod
    def normalize_to_lc_df(cls, df: pd.DataFrame):
        missing_cols = cls.check_columns(df)
        if missing_cols:
            df = cls.add_missing_columns(df, missing_cols=missing_cols)

        lc_data = [LabelCheckerData(series=series) for _, series in df.iterrows()]
        return lc_data

    @staticmethod
    def check_columns(df: pd.DataFrame) -> list:
        return [name for name in column_name_types.keys() if name not in df.columns]

    @staticmethod
    def add_missing_columns(df: pd.DataFrame, missing_cols: list) -> pd.DataFrame:
        new_columns = [*df.columns, *missing_cols]
        return df.reindex(columns=new_columns).convert_dtypes(dtype_backend="pyarrow")
