#!/bin/bash
#PBS -l walltime=02:00:00,mem=190gb,ncpus=48
#PBS -P p66
#PBS -q normal
#PBS -j oe
#PBS -o processed/process_era5.log
#PBS -l wd
#PBS -lstorage=gdata/hh5+gdata/p66

# Process monthly ERA5 grib files into a single netcdf file for each time record.
# Processed files: 103 MB per file, 824 MB per day, 300 GB per year

module use /g/data/xp65/public/modules
module load conda/analysis3-25.08
python3 process_era5.py
