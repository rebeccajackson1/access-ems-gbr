#!/bin/bash
#PBS -l walltime=00:10:00,mem=10gb,ncpus=104
#PBS -P p66
#PBS -q normalsr
#PBS -j oe
#PBS -l other=gdata1
#PBS -lstorage=gdata/p66+gdata/access+gdata/xp65

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

Inputs:
    create_ss_flux.py massflux (kg/gridbox/s) coord_file (list of coords or .shp) template_fname, out_fname 

Optional args:
    --ancil_start_dt  (start date for ancil file)
    --ancil_end_dt    (end date for ancil file)
    --inj_start_dt    (start date for emissions)
    --inj_end_dt      (end date for emissions)
    --inj_start_hr    (start hour for emissions)
    --inj_end_hr      (end hour for emissions)
'

module load python3
module use /g/data/xp65/public/modules
module load conda/analysis3-25.08

cd UM_NS/create_SSflux

### Inputs
massflux=0.03182563e-1
l_pt_src=true
l_periodic=true
coord_file=inputs/Sprayers_Cairns_20km_spacing_coords.csv            # Point sources
#coord_file=inputs/gbr50_region.shp                                  # Even surface flux
out_fpath=out_new
out_fname=gbr_ssflux_20kmSpacing_MedFlux_cairns
template_global=ancils/out/OC_biomass_high_2014_time_slice.nc               # Global
template_rgn=ancils/NE_Aus/4km/ukca/out/OC_biomass_high_2014_time_slice.nc  # NE Aus

### Point source emissions
if $l_pt_src; then       
    if $l_periodic; then
        start_dt=2021-12-30
        end_dt=2022-05-02
        inj_start_dt=2021-12-31
        inj_end_dt=2022-04-30
        inj_start_hr=20 # UTC (6 am UTC+10)
        inj_end_hr=8    # UTC (6 pm UTC+10)
        python3 create_ssflux.py ${massflux} ${coord_file} ${template_global} ${out_fpath}/N216/${out_fname}_tseries.nc --pt_source --ancil_start_dt ${start_dt} --ancil_end_dt ${end_dt} --inj_start_dt ${inj_start_dt} --inj_end_dt ${inj_end_dt} --inj_start_hr ${inj_start_hr} --inj_end_hr ${inj_end_hr}
        python3 create_ssflux.py ${massflux} ${coord_file} ${template_rgn} ${out_fpath}/NE_Aus_4km/${out_fname}_tseries.nc --pt_source --ancil_start_dt ${start_dt} --ancil_end_dt ${end_dt} --inj_start_dt ${inj_start_dt} --inj_end_dt ${inj_end_dt} --inj_start_hr ${inj_start_hr} --inj_end_hr ${inj_end_hr}
    else
        python3 create_ssflux.py ${massflux} ${coord_file} ${template_global} ${out_fpath}/N216/${out_fname}.nc --pt_source 
        python3 create_ssflux.py ${massflux} ${coord_file} ${template_rgn} ${out_fpath}/NE_Aus_4km/${out_fname}.nc --pt_source 
    fi

### Even surface flux over .shp region
else 
    if $l_periodic; then
        start_dt=2021-12-30
        end_dt=2022-05-02
        inj_start_dt=2022-01-01
        inj_end_dt=2022-04-30
        inj_start_hr=20 # UTC (6 am UTC+10)
        inj_end_hr=8    # UTC (6 pm UTC+10)
        python3 create_ssflux.py ${massflux} ${coord_file} ${template_global} ${out_fpath}/n216/${out_fname}_tseries.nc --ancil_start_dt ${start_dt} --ancil_end_dt ${end_dt} --inj_start_dt ${inj_start_dt} --inj_end_dt ${inj_end_dt} --inj_start_hr ${inj_start_hr} --inj_end_hr ${inj_end_hr}
        python3 create_ssflux.py ${massflux} ${coord_file} ${template_rgn} ${out_fpath}/NE_Aus_4km/${out_fname}_tseries.nc --ancil_start_dt ${start_dt} --ancil_end_dt ${end_dt} --inj_start_dt ${inj_start_dt} --inj_end_dt ${inj_end_dt} --inj_start_hr ${inj_start_hr} --inj_end_hr ${inj_end_hr}
    else
        python3 create_ssflux.py ${massflux} ${coord_file} ${template_global} ${out_fpath}/n216/${out_fname}.nc
        python3 create_ssflux.py ${massflux} ${coord_file} ${template_rgn} ${out_fpath}/NE_Aus_4km/${out_fname}.nc 
    fi
fi
