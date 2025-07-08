#!/usr/bin/env python3
"""
ROI File Processor CLI
Processes .roi and .adc files to extract images and generate CSV metadata.
"""

# Usage example:
# python preprocessingIFCB.py /path/to/input_folder --output /path/to/output_folder --verbose

# Expected input folder structure:

# /path/to/input_folder/
# ├── sample1.roi
# ├── sample1.adc
# └── sample1.hdr
# ├── sample2.roi
# ├── sample2.adc
# └── sample2.hdr

import argparse
import sys
from pathlib import Path
import os
import numpy as np
import cv2
import pandas as pd
import uuid
from src.services.preprocessing.BiovolAndSurfaceAreaCalculator.utils.biovol_cal import biovolume

class config():
    area_raio_threshold = 1.2  # Threshold for area ratio
    eccentricity_threshold = 0.5  # Threshold for eccentricity 
    p_threshold = 0.8  # Threshold for p-value in biovolume calculation
    calibration_const = 1.0  # Calibration constant for biovolume calculation

def process_files(input_folder_path, output_folder_path):
    """
    Process ROI and ADC files to extract images and generate CSV metadata.
    
    Args:
        input_folder_path (str): Path to input folder
        output_folder_path (str): Path to output folder
    """
    unique_filenames = list({file.stem for file in Path(input_folder_path).iterdir() if file.is_file()})
    
    for filename in unique_filenames:
        roi_path = os.path.join(input_folder_path, f"{filename}.roi")
        adc_path = os.path.join(input_folder_path, f"{filename}.adc")
        hdr_path = os.path.join(input_folder_path, f"{filename}.hdr")
        
        # Check if required files exist
        if not (Path(roi_path).exists() and Path(adc_path).exists()):
            print(f"Skipping {filename}: missing .roi or .adc file")
            continue
        
        # Create output directory structure
        output_image_folder_path = os.path.join(output_folder_path, filename, filename)
        os.makedirs(output_image_folder_path, exist_ok=True)
        
        # Set up DataFrame columns
        LC_col_str = "Name,Date,Time,CollageFile,ImageFilename,Id,GroupId,Uuid,SrcImage,SrcX,SrcY,ImageX,ImageY,ImageW,ImageH,Timestamp,ElapsedTime,CalConst,CalImage,AbdArea,AbdDiameter,AbdVolume,AspectRatio,AvgBlue,AvgGreen,AvgRed,BiovolumeCylinder,BiovolumePSpheroid,BiovolumeSphere,Ch1Area,Ch1Peak,Ch1Width,Ch2Area,Ch2Ch1Ratio,Ch2Peak,Ch2Width,Ch3Area,Ch3Peak,Ch3Width,CircleFit,Circularity,CircularityHu,Compactness,ConvexPerimeter,Convexity,EdgeGradient,Elongation,EsdDiameter,EsdVolume,FdDiameter,FeretMaxAngle,FeretMinAngle,FiberCurl,FiberStraightness,FilledArea,FilterScore,GeodesicAspectRatio,GeodesicLength,GeodesicThickness,Intensity,Length,Perimeter,Ppc,RatioBlueGreen,RatioRedBlue,RatioRedGreen,Roughness,ScatterArea,ScatterPeak,SigmaIntensity,SphereComplement,SphereCount,SphereUnknown,SphereVolume,SumIntensity,Symmetry,Transparency,Width,BiovolumeMS,SurfaceAreaMS,Preprocessing,PreprocessingTrue,LabelPredicted,ProbabilityScore,LabelTrue"
        LC_columns = LC_col_str.split(",")
        LC_df = pd.DataFrame(columns=LC_columns)
        
        print(f"Processing {filename}...")
        
        # Read ROI data
        roi_data = np.fromfile(roi_path, dtype="uint8")
        
        # Process ADC file
        with open(adc_path, 'r') as adc_data:
            for i, adc_line in enumerate(adc_data):
                print(f"Processed {i} images", end='\r')
                adc_line = adc_line.split(",")
                output_image_name = f"{filename}_{i:05d}.png"
                output_image_path = os.path.join(output_folder_path, filename, filename, output_image_name)
                
                # Populate DataFrame
                LC_df.loc[i, 'Name'] = filename
                LC_df.loc[i, 'ImageFilename'] = output_image_name
                LC_df.loc[i, 'Id'] = i
                LC_df.loc[i, 'GroupId'] = i
                LC_df.loc[i, 'ImageW'] = adc_line[15]
                LC_df.loc[i, 'ImageH'] = adc_line[16]
                LC_df.loc[i, 'Uuid'] = uuid.uuid4().hex.upper()
                LC_df.loc[i, 'Preprocessing'] = 'object'
                
                # Extract and save image
                roi_x = int(adc_line[15])  # ROI width
                roi_y = int(adc_line[16])  # ROI height
                image_size = roi_x * roi_y
                
                if image_size > 0:
                    roi_start_bit = int(adc_line[17])
                    roi_end_bit = roi_start_bit + image_size
                    image = roi_data[roi_start_bit:roi_end_bit].reshape((roi_y, roi_x))

                    _, BiovolumeMS, SurfaceAreaMS = biovolume(image, area_raio_threshold = config.area_raio_threshold, eccentricity_threshold = config.eccentricity_threshold, p_threshold = config.p_threshold, calibration_const=config.calibration_const, debug = False)
                    LC_df.loc[i, 'BiovolumeMS'] = BiovolumeMS
                    LC_df.loc[i, 'SurfaceAreaMS'] = SurfaceAreaMS

                    cv2.imwrite(output_image_path, image)
        
        # Save CSV file
        LC_file_name = f"LabelChecker_{filename}.csv"
        LC_file_path = os.path.join(output_folder_path, filename, LC_file_name)
        LC_df.to_csv(LC_file_path, index=False)
        
        print(f"Completed processing {filename}")


def main():
    parser = argparse.ArgumentParser(description='Process ROI and ADC files to extract images and generate CSV metadata.')
    
    parser.add_argument('input_folder', 
                       help='Path to the input folder containing .roi, .adc, and .hdr files')
    
    parser.add_argument('-o', '--output', 
                       default=None,
                       help='Path to the output folder (default: same as input folder)')
    
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input folder
    input_folder_path = Path(args.input_folder)
    if not input_folder_path.exists():
        print(f"Error: Input folder '{input_folder_path}' does not exist.")
        sys.exit(1)
    
    if not input_folder_path.is_dir():
        print(f"Error: '{input_folder_path}' is not a directory.")
        sys.exit(1)
    
    # Determine output folder
    if args.output:
        output_folder_path = Path(args.output)
    else:
        output_folder_path = input_folder_path
    
    # Create output folder if it doesn't exist
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Input folder: {input_folder_path}")
        print(f"Output folder: {output_folder_path}")
    
    # Process files
    try:
        process_files(str(input_folder_path), str(output_folder_path))
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()