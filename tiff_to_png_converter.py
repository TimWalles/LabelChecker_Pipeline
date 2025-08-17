import os
from pathlib import Path
from PIL import Image
import typer
from rich.progress import track
from typing_extensions import Annotated


def convert_tiff_to_png(input_dir: Path, output_dir: Path = None, delete_original: bool = False) -> None:
    """
    Convert all .tiff files in a directory to .png format.
    
    Args:
        input_dir: Directory containing .tiff files
        output_dir: Directory to save .png files (defaults to input_dir)
        delete_original: Whether to delete original .tiff files after conversion
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .tiff files
    tiff_files = list(input_dir.glob("*.tiff")) + list(input_dir.glob("*.tif"))
    
    if not tiff_files:
        print(f"No .tiff or .tif files found in {input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to convert")
    
    # Convert each file
    for tiff_file in track(tiff_files, description="Converting TIFF to PNG"):
        try:
            # Open the TIFF image
            with Image.open(tiff_file) as img:
                # Create output filename
                png_filename = tiff_file.stem + ".png"
                png_path = output_dir / png_filename
                
                # Convert and save as PNG
                img.save(png_path, "PNG")
                
                # Delete original file if requested
                if delete_original:
                    tiff_file.unlink()
                
        except Exception as e:
            print(f"Error converting {tiff_file.name}: {str(e)}")
            continue
    
    print(f"Conversion complete. PNG files saved to {output_dir}")


def main(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input_dir",
            "-i",
            exists=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Directory containing .tiff files to convert",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output_dir",
            "-o",
            dir_okay=True,
            resolve_path=True,
            help="Directory to save converted .png files (default: same as input)",
        ),
    ] = None,
    delete_original: Annotated[
        bool,
        typer.Option(
            "--delete_original",
            "-d",
            help="Delete original .tiff files after conversion",
        ),
    ] = True,
) -> None:
    """Convert TIFF images to PNG format."""
    convert_tiff_to_png(input_dir, output_dir, delete_original)


if __name__ == "__main__":
    typer.run(main)