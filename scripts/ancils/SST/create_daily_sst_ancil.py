#!/usr/bin/env python3

'''
*********************************************************************************
Create daily SST ancillary files for ACCESS-EMS-GBR
********************************************************************************* 

Creates new SST ancillary files for global and regional domains from daily mean BRAN2020 data, using monthly mean HADISST SST ancils as templates.
Global template ancil is copied from /g/data/access/umdir/ancil/atmos/n216e/orca025/sst/hadisst_6190/v1/qrclim.sst.
NE Aus template ancil is output from the RAS.

BRAN2020 (Bluelink Reanalysis v 2020) is a CSIRO reanalysis dataset that combines model output from the global OFAM model 
(Ocean Forecasting Australia Model, ~0.1 deg resolution near Australia) with observations, using the BODAS 
(Bluelink Ocean Data Assimilation System) data assimilation system.
Reference   : Chamberlain et al. (2021), https://doi.org/10.5194/essd-13-5663-2021
Data use    : https://research.csiro.au/bluelink/outputs/data-access/
Data access : https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f9372_7752_2015_3718
'''

import iris
import mule
import numpy as np
import datetime as dt
from datetime import datetime
import pandas as pd
import xesmf as xe
import xarray as xr
import glob
import argparse

def main(args):
    
    # Inputs:
    ancil_start = datetime.strptime(args.ancil_start, '%Y-%m-%d')
    ancil_end = datetime.strptime(args.ancil_end, '%Y-%m-%d')
    ancil_dir = args.ancil_dir
    ancil_file = args.ancil_fname
    out_dir = args.out_dir
    out_file = args.out_file
    data_dir = args.data_dir
    
    # Load template - monthly climatology:
    ancil_cube = iris.load_cube(ancil_dir+ancil_file)

    # Create time coordinate:
    ndays = int((ancil_end - ancil_start).days)+1
    ancil_time = [ancil_start + dt.timedelta(days = iday) for iday in range(0, ndays)]
    
    # Get file paths and constrain to required dates
    yrs = np.arange(ancil_start.year, ancil_end.year+1)
    files = []
    for iyr in yrs:
        files.extend(sorted(glob.glob(data_dir+'ocean_temp_'+str(iyr)+'*')))
    
    rmv_files = []
    for ifile in files:
        if (int(ifile[-5:-3]) < ancil_start.month) or (int(ifile[-5:-3]) > ancil_end.month):
            rmv_files.append(ifile)
    for ifile in rmv_files:
        files.remove(ifile)

    # Load BRAN2020 data:
    ds = xr.open_mfdataset(files, chunks={"time": "auto"}, concat_dim='Time', combine='nested', decode_timedelta=True).sel(st_ocean=2.5)
    ds['Time'] = ds['Time'].dt.floor('D')
    ds = ds.sel(Time=slice(ancil_start,ancil_end))
    ds = ds.reset_coords()

    # Rename bran_ds time/lat/lon coordinates for use by xe.Regridder
    ds = ds.rename({'Time':'time', 'xt_ocean':'longitude', 'yt_ocean':'latitude'})
    ds = ds.transpose('time','latitude','longitude',...)
    
    # Ensure time-series is complete
    complete_dates = pd.date_range(ancil_start, ancil_end, freq='D') # np.datetime64
    missing_dates = []
    for t in complete_dates:
        if t not in ds.time:
            missing_dates.append(t)

    print(f'Data missing on {len(missing_dates)} days.')
    if len(missing_dates) > 0:
        for i in missing_dates:
            print(i)
        
        # Reindex from partial to complete time-series
        ds = ds.reindex(time=ancil_time)
        ds = ds.ffill(dim='time', limit=30)
        if np.isnan(ds.temp[0,:]).all():
            ds = ds.bfill(dim='time', limit=30)

        for t in range(len(ancil_time)):
            if np.isnan(ds.temp[t,:]).all():
                print(f'WARNING: Missing data after ffill on {ancil_time[t]}.')
    
    # Regrid to ancillary grid
    
    # First convert cube to xr.Dataset
    ancil_da = xr.DataArray.from_iris(ancil_cube)
    ancil_ds = ancil_da.to_dataset(name='ancil_data')
    ancil_ds['mask'] = xr.DataArray(ancil_cube.data.mask, dims=ancil_da.dims)
    
    # Then expand ancil time from monthly to daily
    ancil_ds = ancil_ds.reindex(time=ancil_time)
    ancil_ds.coords['month'] = ancil_ds.time.dt.month
    ancil_ds.coords['dayofyear'] = ancil_ds.time.dt.dayofyear
    for i in range(len(ancil_ds.dayofyear.values)):
        imonth = ancil_ds.month.values[i]
        #constrain_month = iris.Constraint(time = lambda cell: cell.point.month == imonth) # ancil cube dates are not always loaded correctly, use bounds instead
        constrain_month = iris.Constraint(time=lambda cell: cell.bound is not None and cell.bound[0].month == imonth)
        tmp_ancil = ancil_cube.copy().extract(constrain_month)
        ancil_ds.ancil_data[i,:] = tmp_ancil.data
        ancil_ds.mask[i,:] = tmp_ancil.data.mask
         
    # Regrid new SST data using land-sea masks
    '''
    Masks provided to xe.Regridder must be 2D.
    Use adaptive masking for datasets where mask varies temporally (seasonally missing values at high latitudes)
    by setting skipna=True when applying regridder. Output points will be left as NaN when the ratio of missing
    values > na_thres (ie for na_thres=0.01, if >= 1% of values are missing, the cell will be NaN)
    '''
    
    src_grid = xr.where(np.isfinite(ds['temp'][0,:]), 1, 0) # 1 = ocean, 0 = land
    src_grid = src_grid.to_dataset(name='mask')
    src_grid = src_grid.drop_vars('time')

    tgt_grid = xr.where(ancil_ds['mask'][0,:] == 0, 1, 0)
    tgt_grid = tgt_grid.to_dataset(name='mask')
    tgt_grid = tgt_grid.drop_vars('time')

    regridder = xe.Regridder(src_grid, tgt_grid, method='bilinear', extrap_method='nearest_s2d')
    temp = ds['temp'].copy()
    regridded_temp = regridder(temp, skipna=True, na_thres=0.01)
    
    # Convert units from C to K:
    regridded_temp = regridded_temp + 273.15

    # BRAN grid does not extend to 90 deg - manually substitute in default monthly values
    masked_temp = xr.where(regridded_temp.latitude >= np.max(src_grid.latitude), ancil_ds['ancil_data'], regridded_temp)
    masked_temp = xr.where(regridded_temp.latitude <= np.min(src_grid.latitude), ancil_ds['ancil_data'], masked_temp)
    
    # Ensure correct dim order
    ancil_dims = [idim.name() for idim in ancil_cube.dim_coords]
    masked_temp = masked_temp.transpose(ancil_dims[0],ancil_dims[1],ancil_dims[2],...)

    # Save new ancillary file
    # Use Mule - ants.save & iris.save modify grid slightly, which can cause model to crash at recon step.
    # Get template ancil
    src_ancil = mule.ancil.AncilFile.from_file(ancil_dir+ancil_file)
    new_ancil = src_ancil.copy()
    new_data = masked_temp.values # extract values here to make substitution faster

    # Remove existing fields
    for t in range(len(new_ancil.fields)):
        new_ancil.fields.remove(new_ancil.fields[t])
            
    # Add new fields
    for t in range(ndays):
        new_field = src_ancil.fields[0].copy() # template field
        data_provider = mule.ArrayDataProvider(new_data[t,:])
        new_field.set_data_provider(data_provider)

        # Validity time
        new_field.lbyr = ancil_time[t].year
        new_field.lbmon = ancil_time[t].month
        new_field.lbdat = ancil_time[t].day
        new_field.lbhr = ancil_time[t].hour
        new_field.lbmin = ancil_time[t].minute
        new_field.lbsec = ancil_time[t].second

        # Data time
        new_field.lbyrd = ancil_time[t].year
        new_field.lbmond = ancil_time[t].month
        new_field.lbdatd = ancil_time[t].day
        new_field.lbhrd = ancil_time[t].hour
        new_field.lbmind = ancil_time[t].minute 
        new_field.lbsecd = ancil_time[t].second

        # Forecast time (hrs) - difference between validity time & data time
        new_field.lbft = 0
        new_field.lbtim = 1 # frequency of data sampling between lb_ & lb_d (1 hr)
        
        # Put new field into ancil
        new_ancil.fields.append(new_field)

    # Update time metadata
    new_ancil.integer_constants.num_times = len(ancil_time)     # number of times
    new_ancil.fixed_length_header.lookup_dim2 = len(ancil_time) # number of fields (1 per time)
    new_ancil.fixed_length_header.calendar = 1                  # proleptic gregorian calendar
    new_ancil.fixed_length_header.time_type = 1                 # 0=single time, 1=time-series, 2=periodic time-series
        
    # start date
    new_ancil.fixed_length_header.t1_year = ancil_time[0].year
    new_ancil.fixed_length_header.t1_month = ancil_time[0].month
    new_ancil.fixed_length_header.t1_day = ancil_time[0].day
    new_ancil.fixed_length_header.t1_hour = ancil_time[0].hour
    new_ancil.fixed_length_header.t1_minute = ancil_time[0].minute
    new_ancil.fixed_length_header.t1_second = ancil_time[0].second
        
    # end date
    new_ancil.fixed_length_header.t2_year = ancil_time[-1].year
    new_ancil.fixed_length_header.t2_month = ancil_time[-1].month
    new_ancil.fixed_length_header.t2_day = ancil_time[-1].day
    new_ancil.fixed_length_header.t2_hour = ancil_time[-1].hour
    new_ancil.fixed_length_header.t2_minute = ancil_time[-1].minute
    new_ancil.fixed_length_header.t2_second = ancil_time[-1].second

    # Frequency of data - change from monthly to daily
    new_ancil.fixed_length_header.t3_year = 0
    new_ancil.fixed_length_header.t3_month = 0
    new_ancil.fixed_length_header.t3_day = 1
    new_ancil.fixed_length_header.t3_hour = 0
    new_ancil.fixed_length_header.t3_minute = 0
    new_ancil.fixed_length_header.t3_second = 0

    # Save
    new_ancil.to_file(out_dir+out_file)
    print(f'Created daily SST ancillary file: {out_dir}{out_file}\n')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='create_daily_sst_ancil', description="Create daily SST ancillary files for ACCESS-GBR.")
    parser.add_argument("ancil_start", type=str, help="Ancillary start date.")
    parser.add_argument("ancil_end", type=str, help="Ancillary end date.")
    parser.add_argument("ancil_dir", type=str, help="Path to template ancillary files.")
    parser.add_argument("ancil_fname", type=str, help="Template ancillary file name.")
    parser.add_argument("out_dir", type=str, help="Path to save new ancillary file.")
    parser.add_argument("out_file", type=str, help="New ancillary file name.")
    parser.add_argument("data_dir", type=str, help="Path to new source data.")
    args = parser.parse_args()
    
    main(args)    
