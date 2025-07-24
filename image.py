from pathlib import Path
import cv2
from numpy.typing import NDArray

from src.schemas.LabelChecker import LabelCheckerData


# region image utils
def read_image(
    image_path: str,
    grayscale: bool = True,
) -> cv2.typing.MatLike | None:
    return (
        cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if grayscale
        else cv2.imread(image_path, cv2.IMREAD_COLOR)
    )


def crop_image(
    image,
    x_loc: int,
    y_loc: int,
    Width: int,
    height: int,
) -> NDArray:
    return image[y_loc : y_loc + height, x_loc : x_loc + Width]


def get_image(
    image_path: str,
    type_image_collage: bool,
    x_loc: int,
    y_loc: int,
    Width: int,
    height: int,
    grayscale: bool = True,
) -> NDArray:
    image = read_image(
        image_path=image_path,
        grayscale=grayscale,
    )
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if type_image_collage:
        image = crop_image(
            image=image,
            x_loc=x_loc,
            y_loc=y_loc,
            Width=Width,
            height=height,
        )
    return image


def build_image_path(data: LabelCheckerData, directory: Path) -> LabelCheckerData:
    if data.ImageFilename and data.Name:
        data.ImageFilename = Path.joinpath(
            directory, data.Name + " Images", data.ImageFilename
        ).as_posix()
        if not Path(data.ImageFilename).exists():
            raise FileNotFoundError(f"Image file not found: {data.ImageFilename}")
    elif data.CollageFile:
        data.CollageFile = Path.joinpath(directory, data.CollageFile).as_posix()
        if not Path(data.CollageFile).exists():
            raise FileNotFoundError(f"Image file not found: {data.CollageFile}")
    else:
        raise ValueError("Image file not found.")
    return data
