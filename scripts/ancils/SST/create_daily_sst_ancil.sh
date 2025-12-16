#!/bin/bash
#PBS -l walltime=01:00:00,ncpus=104,mem=50gb,wd
#PBS -P p66
#PBS -q normalsr
#PBS -lstorage=gdata/hh5+gdata/p66+scratch/p66+gdata/gb6
#PBS -o create_daily_sst_ancil.out
#PBS -e create_daily_sst_ancil.err

: '
*********************************************************************************
Create daily SST ancillary files for ACCESS-EMS-GBR
********************************************************************************* 

Creates new SST ancillary files for global and nested domains from daily mean BRAN2020 data, using monthly mean HADISST SST ancils as templates.
Global template ancil is copied from /g/data/access/umdir/ancil/atmos/n216e/orca025/sst/hadisst_6190/v1/qrclim.sst.
NE Aus template ancil is output from the RAS.

BRAN2020 (Bluelink Reanalysis v 2020) is a CSIRO reanalysis dataset that combines model output from the global OFAM model 
(Ocean Forecasting Australia Model, ~0.1 deg resolution near Australia) with observations, using the BODAS 
(Bluelink Ocean Data Assimilation System) data assimilation system.
Reference   : Chamberlain et al. (2021), https://doi.org/10.5194/essd-13-5663-2021
Data use    : https://research.csiro.au/bluelink/outputs/data-access/
Data access : https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f9372_7752_2015_3718
'

module use /g/data/xp65/public/modules
module load conda/analysis3-25.08

# Dates
ancil_start=2022-01-01
ancil_end=2022-12-31

# Template ancillary files
ancil_dir=/g/data/p66/rj9627/UM_NS/ancils
ancil_file=qrclim.sst.hadisst

# Output ancillary files
out_dir=/g/data/p66/rj9627/UM_NS/ancils/BRAN2020_SST/out/
out_file=qrclim.sst.bran2020

# New source data
data_dir=/g/data/gb6/BRAN/BRAN2020/daily/

# Run script
cd $out_dir
script_dir=/g/data/p66/rj9627/UM_NS/ancils/BRAN2020_SST/
python3 ${script_dir}create_daily_sst_ancil.py $ancil_start $ancil_end $ancil_dir/global/n216/ $ancil_file $out_dir $out_file.global.n216.${ancil_start:0:4}-${ancil_end:0:4} $data_dir
python3 ${script_dir}create_daily_sst_ancil.py $ancil_start $ancil_end $ancil_dir/NE_Aus/4km/ $ancil_file $out_dir $out_file.NE_Aus.4km.${ancil_start:0:4}-${ancil_end:0:4} $data_dir

exit 0