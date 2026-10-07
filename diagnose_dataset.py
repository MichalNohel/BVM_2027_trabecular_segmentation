#!/usr/bin/env python3
"""
Diagnostic script to verify DICOM dataset structure and find conventional CT series.
"""

import os
from pathlib import Path
from collections import defaultdict
import pydicom
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DatasetDiagnostics:
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.stats = {
            'total_patients': 0,
            'total_dicom_files': 0,
            'conventional_ct_series': 0,
            'other_series': 0,
            'errors': 0,
        }
        self.patients_with_conv_ct = []
        self.series_descriptions = defaultdict(int)
    
    def scan_dataset(self) -> None:
        """Scan the entire dataset and collect statistics."""
        logger.info("="*70)
        logger.info("DATASET DIAGNOSTIC SCAN")
        logger.info("="*70)
        logger.info(f"Source: {self.source_dir}\n")
        
        if not self.source_dir.exists():
            logger.error(f"Directory not found: {self.source_dir}")
            return
        
        # Find all patient folders
        patient_folders = sorted([
            f for f in self.source_dir.iterdir() 
            if f.is_dir() and f.name.startswith('Myel_')
        ])
        
        logger.info(f"Found {len(patient_folders)} patient folders:\n")
        
        self.stats['total_patients'] = len(patient_folders)
        
        for patient_folder in patient_folders:
            self._scan_patient_folder(patient_folder)
        
        self._print_summary()
    
    def _scan_patient_folder(self, patient_folder: Path) -> None:
        """Scan a single patient folder."""
        patient_name = patient_folder.name
        conv_ct_found = False
        series_in_patient = defaultdict(list)
        
        # Recursively find all DICOM files
        for root, dirs, files in os.walk(patient_folder):
            for file in files:
                file_path = Path(root) / file
                
                try:
                    dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                    self.stats['total_dicom_files'] += 1
                    
                    if hasattr(dcm, 'SeriesDescription'):
                        series_desc = str(dcm.SeriesDescription)
                        self.series_descriptions[series_desc] += 1
                        series_in_patient[series_desc].append(file_path)
                        
                        if 'konv' in series_desc.lower():
                            conv_ct_found = True
                            self.stats['conventional_ct_series'] += 1
                except Exception as e:
                    self.stats['errors'] += 1
        
        # Print patient info
        if conv_ct_found:
            self.patients_with_conv_ct.append(patient_name)
            logger.info(f"✓ {patient_name}: {len(series_in_patient)} series")
            for series_desc, files in sorted(series_in_patient.items()):
                marker = "🔍 KONV" if 'konv' in series_desc.lower() else "  "
                logger.info(f"    {marker} [{len(files):3d} files] {series_desc}")
        else:
            logger.info(f"✗ {patient_name}: {len(series_in_patient)} series (NO CONVENTIONAL CT)")
            for series_desc, files in sorted(series_in_patient.items()):
                logger.info(f"      [{len(files):3d} files] {series_desc}")
    
    def _print_summary(self) -> None:
        """Print comprehensive summary."""
        logger.info("\n" + "="*70)
        logger.info("SUMMARY")
        logger.info("="*70)
        logger.info(f"Total patient folders: {self.stats['total_patients']}")
        logger.info(f"Patients with conventional CT: {len(self.patients_with_conv_ct)}")
        logger.info(f"Total DICOM files found: {self.stats['total_dicom_files']}")
        logger.info(f"Conventional CT series found: {self.stats['conventional_ct_series']}")
        logger.info(f"Read errors: {self.stats['errors']}")
        
        logger.info("\n" + "="*70)
        logger.info("ALL SERIES DESCRIPTIONS FOUND")
        logger.info("="*70)
        for desc in sorted(self.series_descriptions.keys()):
            count = self.series_descriptions[desc]
            marker = "✓ WILL BE EXTRACTED" if 'konv' in desc.lower() else ""
            logger.info(f"[{count:4d}x] {desc} {marker}")
        
        logger.info("\n" + "="*70)
        logger.info("PATIENTS WITH CONVENTIONAL CT (Ready for extraction)")
        logger.info("="*70)
        for patient in sorted(self.patients_with_conv_ct):
            logger.info(f"  • {patient}")
        
        logger.info("\n" + "="*70)
        logger.info("RECOMMENDATION")
        logger.info("="*70)
        if len(self.patients_with_conv_ct) > 0:
            logger.info(f"✓ Dataset ready! {len(self.patients_with_conv_ct)}/{self.stats['total_patients']} "
                        f"patients have conventional CT.")
            logger.info("  Run: python extract_conventional_ct.py")
        else:
            logger.warning("✗ No conventional CT ('konv') series found!")
            logger.warning("  Please verify the dataset structure and DICOM tags.")
        
        logger.info("="*70 + "\n")


def main():
    source_dir = r"F:\TCIA\manifest-1774389300184\Spinal-Multiple-Myeloma-SEG"
    
    diagnostics = DatasetDiagnostics(source_dir)
    diagnostics.scan_dataset()


if __name__ == "__main__":
    main()
