from rich import print
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

from src.schemas.ModelConfig import ModelConfig
from src.logging import log_error


class Dataloader:
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_dataloader(
        self,
        df: pd.DataFrame,
    ) -> tf.data.Dataset | None:

        try:
            self.check_features(columns=df.columns)
            ds = tf.data.Dataset.from_tensor_slices(dict(df))
            ds = ds.map(self.get_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.batch(22).prefetch(tf.data.experimental.AUTOTUNE)

            return ds
        except Exception as e:
            log_error(message=str(e))
            return None

    def get_data(self, row: pd.Series):
        image = self.get_image(row)
        features = self.get_features(row)
        return [(features, image)]

    # region image processing
    def get_image(self, row: pd.Series):
        image = self.decode_image(row=row)
        return self.resize_image(image)

    def decode_image(self, row: pd.Series):
        if "ImageFilename" in row:
            image_string = tf.io.read_file(row["ImageFilename"])
            image = tf.io.decode_png(image_string, channels=3)
            return image
        else:
            image_string = tf.io.read_file(row["CollageFile"])
            image = tfio.experimental.image.decode_tiff(image_string)
            image = self.remove_alpha_channel(image)
            image = self.crop_image(row, image)
            return image

    def remove_alpha_channel(self, image):
        return tf.convert_to_tensor(
            image[:, :, : self.model_config.Input_shape[2]]
        )  # remove alpha channel

    @staticmethod
    def crop_image(row: pd.Series, image):
        image_x = tf.squeeze(row["ImageX"])
        image_y = tf.squeeze(row["ImageY"])
        image_width = tf.squeeze(row["ImageW"])
        image_height = tf.squeeze(row["ImageH"])
        return image[
            int(image_y) : int(image_y) + int(image_height),
            int(image_x) : int(image_x) + int(image_width),
        ]

    def resize_image(self, image):
        image = tf.image.resize(image, self.model_config.Input_shape[:2])  # H, W only
        return image

    # endregion

    # region get features
    def get_features(self, row: pd.Series):
        return tf.convert_to_tensor(
            [row[feature] for feature in self.model_config.Features], dtype=tf.float64
        )

    # endregion

    # region helper methods
    def check_features(self, columns: pd.Index):
        if not set(self.model_config.Features).issubset(set(columns)):
            raise ValueError(
                f"{[feature for feature in self.model_config.Features if feature not in columns]} features not found in the data"
            )
        return columns

    # endregion
