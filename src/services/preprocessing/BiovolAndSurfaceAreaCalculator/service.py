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
            message=f"""Running [bold magenta]biovol and surface area calculator[/bold magenta] with settings:
        area_raio_threshold value: [bold magenta]{cls.config.area_raio_threshold}[/bold magenta]
        eccentricity_threshold value: [bold magenta]{cls.config.eccentricity_threshold}[/bold magenta]
        p_threshold value: [bold magenta]{cls.config.p_threshold}[/bold magenta]""",
            verbose=verbose,
        )
    
        # Get the version of VisualSpreadsheet from summary files
        # summary_files = list(data_directory.glob("*summary.csv"))
        # if summary_files:
        #     print(f"Version: {get_visualspreadsheet_version(summary_files[0])}")

        # update service settings if set for saving
        if service_settings:
            service_settings.update({cls.__name__: subset_settings(cls.config.model_dump(), cls.__name__)})

        return [
            bcal_function(lc_data=lc_data, data_directory=data_directory, config=cls.config)
            for lc_data in data
        ], service_settings

def bcal_function(lc_data: LabelCheckerData, data_directory, config: Config) -> LabelCheckerData:
    CollageFile_name = lc_data.get_value("CollageFile")
    calibration_const = lc_data.get_value("CalConst")

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

    _, lc_data.BiovolumeMS, lc_data.SurfaceAreaMS = biovolume(image, area_raio_threshold = config.area_raio_threshold, eccentricity_threshold = config.eccentricity_threshold, p_threshold = config.p_threshold, calibration_const=calibration_const, debug = False)

    return lc_data

def get_visualspreadsheet_version(filename):
    import re
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if 'Software' in line:
                software_string = line.split(',')[1].strip()
                match = re.search(r'VisualSpreadsheet(\d+)', software_string)
                return match.group(1) if match else None
    return None
