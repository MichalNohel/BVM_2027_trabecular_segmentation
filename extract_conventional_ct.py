#!/usr/bin/env python3
"""
Script to extract conventional CT DICOM files from TCIA dataset and convert to NIfTI.
Uses pydicom for metadata reading, SimpleITK for DICOM series reading and NIfTI writing.
Filters DICOM files by tag 0008.103E containing "konv" pattern.
Handles multiple follow-up scans (e.g., Myel_12_a, Myel_12_b).
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import SimpleITK as sitk
import pydicom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DICOMProcessor:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
    
    def dicom_to_nifti_sitk(self, dicom_files: List[Path], output_path: Path, series_desc: str) -> bool:
        """Convert DICOM series to NIfTI using SimpleITK."""
        try:
            if not dicom_files or len(dicom_files) == 0:
                logger.warning(f"No DICOM files provided for {series_desc}")
                return False
            
            # Sort DICOM files by Instance Number
            sorted_files = []
            for dcm_file in dicom_files:
                try:
                    dcm = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
                    instance_num = int(dcm.InstanceNumber) if hasattr(dcm, 'InstanceNumber') else 0
                    sorted_files.append((instance_num, str(dcm_file)))
                except Exception as e:
                    logger.warning(f"Could not read {dcm_file}: {e}")
            
            if not sorted_files:
                logger.warning(f"No valid DICOM files for {series_desc}")
                return False
            
            # Sort by instance number
            sorted_files.sort(key=lambda x: x[0])
            file_paths = [str(f[1]) for f in sorted_files]
            
            logger.info(f"  Reading {len(file_paths)} DICOM files with SITK...")
            
            # Use SITK to read DICOM series
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_paths)
            
            try:
                image = reader.Execute()
                
                if image is None or image.GetNumberOfPixels() == 0:
                    logger.warning(f"Failed to read DICOM series {series_desc}")
                    return False
                
                # Log image info
                size = image.GetSize()
                spacing = image.GetSpacing()
                
                logger.info(f"  Size: {size} (Z={size[2]}, Y={size[1]}, X={size[0]})")
                logger.info(f"  Spacing: {spacing}")
                
                # Negate Z-axis in direction matrix
                direction = list(image.GetDirection())
                direction[8] *= -1  # [2,2] element for Z-axis
                image.SetDirection(direction)
                
                # Cast to float32
                image = sitk.Cast(image, sitk.sitkFloat32)
                
                # Write NIfTI
                writer = sitk.ImageFileWriter()
                writer.SetFileName(str(output_path))
                writer.Execute(image)
                
                logger.info(f"  Saved: {output_path.name}")
                return True
                
            except Exception as e:
                logger.error(f"Error reading DICOM series: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in dicom_to_nifti_sitk: {e}")
            return False
    
    def process_patient_folder(self, patient_folder: Path) -> Tuple[int, int]:
        """Process a single patient folder (handles both simple and multi-scan cases)."""
        converted = 0
        skipped = 0
        
        patient_name = patient_folder.name
        logger.info(f"Processing: {patient_name}")
        
        # Collect DICOM files by series description
        dicom_by_series = {}  # series_description -> list of file paths
        
        # Walk through all files in patient folder
        for root, dirs, files in os.walk(patient_folder):
            for file in files:
                file_path = Path(root) / file
                
                try:
                    # Read DICOM metadata
                    dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                    
                    # Check for SeriesDescription tag
                    if hasattr(dcm, 'SeriesDescription'):
                        series_desc = str(dcm.SeriesDescription)
                        
                        # Filter for conventional CT (_konv)
                        if '_konv' in series_desc.lower():
                            if series_desc not in dicom_by_series:
                                dicom_by_series[series_desc] = []
                            dicom_by_series[series_desc].append(file_path)
                
                except Exception as e:
                    # Not a DICOM file, skip silently
                    pass
        
        # Convert each series found
        for series_desc, file_list in sorted(dicom_by_series.items()):
            try:
                logger.info(f"  Series: {series_desc} ({len(file_list)} files)")
                
                output_filename = f"{series_desc}.nii.gz"
                output_path = self.output_dir / output_filename
                
                if self.dicom_to_nifti_sitk(file_list, output_path, series_desc):
                    converted += 1
                    self.processed_count += 1
                else:
                    skipped += 1
                    self.skipped_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing series {series_desc}: {e}")
                self.error_count += 1
        
        if not dicom_by_series:
            logger.warning(f"  No conventional CT found")
            skipped += 1
            self.skipped_count += 1
        
        return converted, skipped
    
    def run(self) -> None:
        """Main processing loop."""
        logger.info(f"Source directory: {self.source_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Starting extraction of conventional CT DICOM files...\n")
        
        if not self.source_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_dir}")
            return
        
        # Find all patient folders (Myel_XXX, including _a, _b variants)
        patient_folders = sorted([
            f for f in self.source_dir.iterdir() 
            if f.is_dir() and f.name.startswith('Myel_')
        ])
        
        logger.info(f"Found {len(patient_folders)} patient folders")
        
        total_converted = 0
        total_skipped = 0
        
        for patient_folder in patient_folders:
            try:
                converted, skipped = self.process_patient_folder(patient_folder)
                total_converted += converted
                total_skipped += skipped
            except Exception as e:
                logger.error(f"Error processing folder {patient_folder.name}: {e}")
                self.error_count += 1
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Successfully converted: {self.processed_count}")
        logger.info(f"Skipped: {self.skipped_count}")
        logger.info(f"Errors: {self.error_count}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*60)


def main():
    # Configuration
    source_dir = r"F:\TCIA\manifest-1774389300184\Spinal-Multiple-Myeloma-SEG"
    output_dir = r"F:\BVM_2027_DATA\Spinal-Multiple-Myeloma-SEG"
    
    # Verify source exists
    if not Path(source_dir).exists():
        logger.error(f"Source directory not found: {source_dir}")
        return
    
    # Create processor and run
    processor = DICOMProcessor(source_dir, output_dir)
    processor.run()


if __name__ == "__main__":
    main()
