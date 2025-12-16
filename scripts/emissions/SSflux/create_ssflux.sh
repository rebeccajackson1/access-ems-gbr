#!/bin/bash

: '
*********************************************************************************
Script to create sea salt emissions ancillary file
*********************************************************************************

Choose emitted sea salt mass, timing (periodic or constant) and locations
(point sources or assume an even surface flux over an area).

Test mass fluxes (kg/gridbox/s)
    Low  : 0.03182563e-2 (0.32 g/s)
    Med  : 0.03182563e-1 (3.18 g/s)
    High : 0.03182563 (31.83 g/s)
'

module use /g/data/xp65/public/modules
module load conda/analysis3-25.07
cd /g/data/p66/rj9627/UM_NS/create_SSflux

massflux=0.03182563

# Periodic point-source emissions
# -------------------------------
coord_file=/g/data/p66/rj9627/UM_NS/create_SSflux/inputs/Sprayers_Cairns_50km_spacing_coords.csv # Point source emissions
out_fname=gbr_ssflux_50kmSpacing_HighFlux_cairns50_tseries.nc
start_dt=2021-12-30
end_dt=2022-05-02
inj_start_dt=2022-01-01
inj_end_dt=2022-04-30
inj_start_hr=20 # UTC (6 am UTC+10)
inj_end_hr=8    # UTC (6 pm UTC+10)
#template_fname=/g/data/access/projects/access/data/ukca/RNS/ancils/out/OC_biomass_high_2014_time_slice.nc # Global
#out_fpath=out/N216/${out_fname}
template_fname=/g/data/p66/rj9627/UM_NS/ancils/NE_Aus/4km/ukca/out/OC_biomass_high_2014_time_slice.nc      # NE Aus
#out_fpath=out/NE_Aus_4km/${out_fname}
out_fpath=test/NE_Aus_4km/${out_fname}
python3 create_ssflux.py ${massflux} ${coord_file} ${template_fname} ${out_fpath} --pt_source --ancil_start_dt ${start_dt} --ancil_end_dt ${end_dt} --inj_start_dt ${inj_start_dt} --inj_end_dt ${inj_end_dt} --inj_start_hr ${inj_start_hr} --inj_end_hr ${inj_end_hr}

# Constant emissions assuming an even surface flux
# ------------------------------------------------
#coord_file=/g/data/p66/rj9627/UM_NS/create_SSflux/inputs/gbr50_region.shp
#out_fname=gbr_ssflux_20kmSpacing_HighFlux_gbr50_constant.nc
#template_fname=/g/data/access/projects/access/data/ukca/RNS/ancils/out/OC_biomass_high_2014_time_slice.nc # Global
#out_fpath=out/N216/${out_fname}
#template_fname=/g/data/p66/rj9627/UM_NS/ancils/NE_Aus/4km/ukca/out/OC_biomass_high_2014_time_slice.nc     # NE Aus
#out_fpath=out/NE_Aus_4km/${out_fname}
#python3 create_ssflux.py ${massflux}  ${coord_file} ${template_fname} ${out_fpath}
