#!/usr/bin/env python3

'''
Create sea salt emissions file for RRAP MCB scenarios.

Sea salt mass flux, location(s) and timing are specified in a netcdf emissions file (add to ukca_ems).
The emission mode mean dry diameter, GSD and mode fraction are specified in an additional code branch:
https://code.metoffice.gov.uk/trac/ukca/browser/main/branches/dev/matthewwoodhouse/um13.0_um13.0_added_SS

Uses the xp65 conda/analysis3-25.08 environment on NCI.
'''

import iris
import iris.analysis
from iris.analysis.cartography import area_weights
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd
import regionmask
import xarray as xr
import argparse
import warnings
warnings.simplefilter("ignore")

def main(args):
    
    # Inputs:
    massflux = args.massflux
    coord_file = args.coord_file
    template_fname = args.template_fname
    out_fname = args.out_fname
    pt_source = args.pt_source
    ancil_start_dt = args.ancil_start_dt
    ancil_end_dt = args.ancil_end_dt
    inj_start_dt = args.inj_start_dt
    inj_end_dt = args.inj_end_dt
    inj_start_hr = args.inj_start_hr
    inj_end_hr = args.inj_end_hr
    
    # Emission type:
    if (pt_source is None):
        pt_source = False
    
    # Set ancillary file dates:
    if (ancil_start_dt is not None) and (ancil_end_dt is not None):
        ancil_start_dt = dt.datetime(int(ancil_start_dt[0:4]), int(ancil_start_dt[5:7]), int(ancil_start_dt[8:10]))
        ancil_end_dt = dt.datetime(int(ancil_end_dt[0:4]), int(ancil_end_dt[5:7]), int(ancil_end_dt[8:10]))
        timing = 'periodic'
    else:
        timing = 'constant'
    
    # Set sea salt emissions dates:
    if (inj_start_dt is not None) and (inj_end_dt is not None):
        if timing == 'constant':
            print('ancil_start/end not provided. Assuming constant emissions.')
        else:
            inj_start_dt = dt.datetime(int(inj_start_dt[0:4]), int(inj_start_dt[5:7]), int(inj_start_dt[8:10]))
            inj_end_dt = dt.datetime(int(inj_end_dt[0:4]), int(inj_end_dt[5:7]), int(inj_end_dt[8:10]))
        
    # Set sea salt emission timing (hourly frequency):
    if (inj_start_hr is not None) and (inj_end_hr is not None) and (timing == 'constant'):
        print('ancil_start/end not provided. Assuming constant emissions.')
        
    print(f'Creating SS emissions file:')
    print(f'    Mass flux:  {massflux} kg/gridbox/s')
    print(f'    Timing:     {timing}')
    if timing != 'constant':
        print(f'    File dates: {ancil_start_dt} - {ancil_end_dt}')
        print(f'    Inj dates:  {inj_start_dt} - {inj_end_dt}')
        print(f'    Inj hours:  {inj_start_hr} - {inj_end_hr} (UTC)')
        
    # Get template cube
    cube = iris.load_cube(template_fname)
    template_cube = cube[0:1,:,:,:].copy() # single datetime - expanded if timing is periodic 
    template_cube.data[:] = 0.0 
    out_cube = template_cube.copy()

    # Handle emission timing
    if timing == 'periodic':
        
        # File dates:
        ndays = int((ancil_end_dt - ancil_start_dt).days) + 1
        ancil_time = [ancil_start_dt + dt.timedelta(hours=t) for t in range(ndays * 24)] # datetime
        ancil_time_fmt = [(ancil_time[t] - dt.datetime(1850,1,1)).days + (ancil_time[t] - dt.datetime(1850,1,1)).seconds/60/60/24 for t in range(ndays*24)] # days since 1850-01-01 00:00:00
        
        # SS emissions timing:
        n_injdays = int((inj_end_dt - inj_start_dt).days) + 1
        emiss_hrs = []
        if inj_end_hr < inj_start_hr:
            emiss_hrs.extend(np.arange(inj_start_hr, 23+1))
            emiss_hrs.extend(np.arange(0, inj_end_hr+1))
        else:
            emiss_hrs = np.arange(inj_start_hr, inj_end_hr+1) 

        emiss_dates = []
        for iday in range(0,n_injdays):
            for ihr in emiss_hrs:
                if (ihr < inj_start_hr):
                    emiss_dates.append(inj_start_dt + dt.timedelta(days=1) + dt.timedelta(days=iday, hours=int(ihr)))
                else:
                    emiss_dates.append(inj_start_dt + dt.timedelta(days=iday, hours=int(ihr)))
                    
        # Time index when SS is emitted:
        emiss_idx = []
        for t in emiss_dates:
            emiss_idx.append(ancil_time.index(t))

        # Expand cube time coord:
        out_cube = out_cube.interpolate([('time', ancil_time_fmt)], iris.analysis.Nearest()) # expand time coordinate
        
    # Add SS mass flux (converted from kg/gridbox/s --> kg/m2/s):
    grid_area = iris.analysis.cartography.area_weights(out_cube[0,0,:,:])
    
    if pt_source == False: # even surface flux over shapefile area
        cube_grid = xr.DataArray.from_iris(out_cube)
        shp = gpd.read_file(coord_file)
        shp_mask = regionmask.mask_geopandas(shp, cube_grid.longitude, cube_grid.latitude)
        massflux_2d = np.zeros(np.shape(shp_mask))
        massflux_2d[:,:] = massflux
        mass_flux = massflux_2d / grid_area
        mass_flux = xr.where(shp_mask == 0, mass_flux, 0.0, keep_attrs=True)
        if timing == 'constant':
            out_cube.data[:] = mass_flux.values
        else:
            out_cube.data[emiss_idx,:] = mass_flux.values
                
    else: # point source emissions
        inj_coords = pd.read_csv(coord_file, sep=',')
        inj_lats = np.asarray(inj_coords.Lat)
        inj_lons = np.asarray(inj_coords.Lon)
        nlocations = len(inj_lats)
        cube_lats = out_cube.coord('latitude').points
        cube_lons = out_cube.coord('longitude').points
        for iloc in range(nlocations):
            ilat = np.abs(cube_lats - inj_lats[iloc]).argmin() # closest gridbox
            ilon = np.abs(cube_lons - inj_lons[iloc]).argmin()
            imass = massflux / grid_area[ilat,ilon]
            if timing == 'constant':
                out_cube.data[:,:,ilat,ilon] = imass
            else:
                out_cube.data[emiss_idx,:,ilat,ilon] = imass
    
    # Set variable attributes:
    out_cube.var_name = 'SS_added' 
    out_cube.long_name = 'Additional primary SS emissions'
    out_cube.standard_name = ''
    out_cube.units = 'kg m-2 s-1'
    out_cube.lowest_level = 1 # surface

    # Set global attributes:
    out_cube.attributes['tracer_name'] = 'SS_added' # add to aero_ems_species (ukca_emiss_mode_mod.F90)
    out_cube.attributes['vertical_scaling'] = 'surface'
    out_cube.attributes['update_freq_in_hours'] = '1'
    if timing == 'constant':
        out_cube.attributes['update_type'] = '0' # 0 = single time; 1 = time-series; 2 = 12-month climatology
    else:
        out_cube.attributes['update_type'] = '1'

    # Save file
    var_atts = ['tracer_name', 'vertical_scaling', 'highest_level', 'lowest_level']
    iris.save(out_cube, out_fname, local_keys = var_atts)
    print(f'Saved file: {out_fname}')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='create_ssflux', description="Create sea salt emission file for RRAP MCB scenarios.")

    parser.add_argument("massflux", type=float, help="Total sea salt mass flux per gridbox.")
    parser.add_argument("coord_file", type=str, help="CSV file with a list of injection coords (if -pt_source flag is used), or shapefile for emissionarea.")
    parser.add_argument("template_fname", type=str, help="Emission ancillary file to use as template.")
    parser.add_argument("out_fname", type=str, help="Out file name.")
    parser.add_argument("--pt_source", action='store_true', help="Use this flag for point source emissions. Default assumes an even surface flux.")
    parser.add_argument("--ancil_start_dt", type=str, required=False, help="File start date (use to create time-series if not constant emissions).")
    parser.add_argument("--ancil_end_dt", type=str, required=False, help="File end date.")
    parser.add_argument("--inj_start_dt", type=str, required=False, help="Date to start emitting (if not constant emissions).")
    parser.add_argument("--inj_end_dt", type=str, required=False, help="Date to stop emitting.")
    parser.add_argument("--inj_start_hr", type=int, required=False, help="Hour to start emitting in UTC (if not constant emissions).")
    parser.add_argument("--inj_end_hr", type=int, required=False, help="Hour to stop emitting in UTC.")
    
    args = parser.parse_args()
    
    main(args)

