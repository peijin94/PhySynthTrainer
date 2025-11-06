#!/usr/bin/env python3
"""
Collect background from FITS files.
For each FITS file with shape (1, 1, 731, n_time), compute the median along the time axis
to get a [731] array. Combine all files into an N x 731 array and save as NPZ.
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count

def process_fits_file(fits_path):
    """
    Process a single FITS file and return the median spectrum.
    
    Args:
        fits_path: Path to FITS file (as string)
        
    Returns:
        tuple: (filename, median_spectrum) or (filename, None) on error
    """
    try:
        with fits.open(fits_path, memmap=True, lazy_load_hdus=True) as hdul:
            data = hdul[0].data  # Shape: (1, 1, 731, n_time)
            
            # Remove singleton dimensions and compute median along time axis
            data_2d = data[0, 0, :, :]  # Shape: (731, n_time)
            median_spectrum = np.nanmedian(data_2d, axis=1)  # Shape: (731,)
            
            return (Path(fits_path).name, median_spectrum)
    except Exception as e:
        print(f"Error processing {fits_path}: {e}", file=sys.stderr)
        return (Path(fits_path).name, None)

def main():
    # Directory containing FITS files
    fits_dir = Path("/nas7a/beam/fits_v1/fits/2024")
    
    # Get all FITS files
    fits_files = sorted(fits_dir.glob("*.fits"))
    print(f"Found {len(fits_files)} FITS files")
    
    if len(fits_files) == 0:
        print("No FITS files found!")
        return
    
    # Convert to strings for multiprocessing
    fits_paths = [str(f) for f in fits_files]
    
    # Determine number of processes (use fewer cores to avoid memory issues)
    n_processes = min(8, cpu_count())
    print(f"Using {n_processes} parallel processes")
    
    # Process all files in parallel
    median_spectra = []
    filenames = []
    
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_fits_file, fits_paths),
            total=len(fits_paths),
            desc="Processing FITS files"
        ))
    
    # Collect results
    for filename, median_spectrum in results:
        if median_spectrum is not None:
            median_spectra.append(median_spectrum)
            filenames.append(filename)
    
    # Convert to numpy array
    median_spectra = np.array(median_spectra)  # Shape: (N, 731)
    
    print(f"\nProcessed {len(median_spectra)} files successfully")
    print(f"Final array shape: {median_spectra.shape}")
    
    # Save to NPZ file
    output_path = Path("./background_2024.npz")
    np.savez(
        output_path,
        background=median_spectra,
        filenames=filenames,
        description="Median background spectra from 2024 FITS files. Shape: (N, 731)"
    )
    
    print(f"\nSaved background data to: {output_path}")
    print(f"  - background: shape {median_spectra.shape}")
    print(f"  - filenames: {len(filenames)} entries")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Mean of all medians: {np.mean(median_spectra):.4f}")
    print(f"  Std of all medians: {np.std(median_spectra):.4f}")
    print(f"  Min: {np.min(median_spectra):.4f}")
    print(f"  Max: {np.max(median_spectra):.4f}")

if __name__ == "__main__":
    main()
