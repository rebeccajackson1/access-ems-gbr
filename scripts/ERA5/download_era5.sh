#!/bin/bash
#PBS -l walltime=10:00:00,mem=190gb
#PBS -P p66
#PBS -q copyq
#PBS -j oe
#PBS -o downloads/download_era5.log
#PBS -l wd
#PBS -lstorage=gdata/hh5+gdata/p66
#PBS -v PYTHONPATH=/home/578/rj9627/.local/lib/python3.10/site-packages

# 1. Download ERA5 nudging data as a single grib file for each month (fastest) (download_era5.py)
# 2. Process into a single file for each time record (process_era5.py)

# 1 month takes ~1.5 hours to request then download and is ~14 GB.

module use /g/data/xp65/public/modules
module load conda/analysis3-25.08
python3 download_era5_monthly.py
