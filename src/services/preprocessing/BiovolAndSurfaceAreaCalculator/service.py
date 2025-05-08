from typing import Tuple
from pathlib import Path
import os

from src.enums.LabelChecker import LabelChecker
from src.logging import log_error, log_info
from src.schemas.LabelChecker import LabelCheckerData
from src.services.config import Config
from src.utils.ServiceSettings import ServiceSettings, subset_settings

from .utils.biovol_cal import biovolume, get_nparray_from_tiff, get_nparray_from_png

class BiovolAndSurfaceAreaCalculator:
    config = Config.preprocessing

    @classmethod
    def process_data(
        cls,
        data: list[LabelCheckerData],
        data_directory: Path,
        verbose: bool,
        service_settings: ServiceSettings | None,
    ) -> Tuple[list[LabelCheckerData], ServiceSettings | None]:
        if not cls.config.biovol_and_surface_area_calculator_active:
            log_info(
                message=f"[bold magenta]biovol and surface area calculator[/bold magenta] switched [bold red]off[/bold red].",
                verbose=verbose,
            )
            if service_settings:
                service_settings.remove(cls.__name__)
            return (data, service_settings)

        log_info(
            message=f"""Running [bold magenta]biovol and surface area calculator[/bold magenta]""",
            verbose=verbose,
        )

        # update service settings if set for saving
        if service_settings:
            service_settings.update({cls.__name__: subset_settings(cls.config.model_dump(), cls.__name__)})

        return [
            bcal_function(lc_data=lc_data, data_directory=data_directory)
            for lc_data in data
        ], service_settings


def bcal_function(lc_data: LabelCheckerData, data_directory) -> LabelCheckerData:

    CollageFile_name = lc_data.get_value("CollageFile")

    if CollageFile_name:
        x = lc_data.get_value("ImageX")
        y = lc_data.get_value("ImageY") 
        w = lc_data.get_value("ImageW")
        h = lc_data.get_value("ImageH")
        CollageFile_name = lc_data.get_value("CollageFile")

        CollageFile_path = os.path.join(data_directory, CollageFile_name)

        image = get_nparray_from_tiff(CollageFile_path, x, y, w, h)

    else:
        ImageFilename = lc_data.get_value("ImageFilename")
        Sample_name = lc_data.get_value("Name")
        
        ImageFile_path = os.path.join(data_directory, Sample_name, ImageFilename)

        image = get_nparray_from_png(ImageFile_path)

    _, lc_data.BiovolumeHSosik, lc_data.SurfaceAreaHSosik = biovolume(image)

    return lc_data
