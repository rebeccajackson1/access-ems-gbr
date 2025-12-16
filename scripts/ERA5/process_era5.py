# Process ERA5 data 

# Files must be formatted as follows:
# Single time record, provided in name as (prefix)YYYYMMDDHH(suffix)
# w. variables 'T' (temp), 'U' (u wind), 'V' (v wind), 'LNSP' (log surface pressure) 
# and dimensions 'time', 'hybrid', 'latitude', 'longitude'

# 103 MB per file, 824 MB per day, 300 GB per year

import datetime as dt
import xarray as xr
import numpy as np
import xesmf as xe
import cfgrib

start_date = dt.datetime(2015,1,1)
end_date = dt.datetime(2015,1,31)
ndays = (end_date - start_date).days + 1

date_list = [start_date + dt.timedelta(days=t) for t in range(ndays)]
t_hr = [hr*3 for hr in range(int(24/3))]

for t in range(len(date_list)):
    
    # Load file:
    in_file = "/g/data/p66/rj9627/ERA5/downloads/era5_1deg_ml137_"+date_list[t].strftime('%Y')+date_list[t].strftime('%m')+"_download.grib"
    
    #ds_in = xr.open_dataset(in_file, decode_times=False, engine='cfgrib')
    ds_tuv, ds_lnsp = cfgrib.open_datasets(in_file, decode_times=False)
    
    ds_tuv = ds_tuv.reindex(latitude=list(reversed(ds_tuv.latitude)))
    ds_lnsp = ds_lnsp.reindex(latitude=list(reversed(ds_lnsp.latitude)))
        
    # Get coordinates:
    longitude = xr.DataArray(ds_tuv['longitude'].values, dims=['longitude'])
    latitude = xr.DataArray(ds_tuv['latitude'].values, dims=['latitude'])
    hybrid = xr.DataArray(np.float32(ds_tuv['hybrid'].values), dims=['hybrid'])
    hybrid_1 = xr.DataArray(np.float32(ds_tuv['hybrid'].isel(hybrid=[136]).values), dims=['hybrid_1'])

    # Latitude and longitude offset by 1/2 grid length 
    longitude_1 = xr.DataArray(ds_tuv['longitude'].sel(longitude=slice(0,359)).values + 0.5, dims=['longitude_1'])
    latitude_1 = xr.DataArray(ds_tuv['latitude'].sel(latitude=slice(-90,89)).values + 0.5, dims=['latitude_1'])

    grid_in = xr.Dataset({'latitude' : (['latitude'], latitude.values),
                          'longitude' : (['longitude'], longitude.values)})
    
    ugrid_out = xr.Dataset({'latitude' : (['latitude'], ds_tuv['latitude'].values),
                            'longitude' : (['longitude'], longitude_1.values)})
     
    u_regridder = xe.Regridder(grid_in, ugrid_out, method='bilinear', periodic=True)
    
    vgrid_out = xr.Dataset({'latitude' : (['latitude'], latitude_1.values),
                            'longitude' : (['longitude'], ds_tuv['longitude'].values)})
        
    v_regridder = xe.Regridder(grid_in, vgrid_out, method='bilinear', periodic=True)
    
    # Build new dataset for each hour:
    for hr in range(len(t_hr)):
        
        time_dt = date_list[t] + dt.timedelta(hours=t_hr[hr])
        time_grib = ((time_dt - dt.datetime(1970,1,1)).days*24*60*60) + ((time_dt - dt.datetime(1970,1,1)).seconds) # grib file: seconds since 1970,1,1
        time_out = ((time_dt - dt.datetime(1900,1,1)).days*24) + ((time_dt - dt.datetime(1900,1,1)).seconds/60/60) # nudging data: hours since 1900,1,1
        
        ds_out = xr.Dataset({'time' : np.float32(ds_tuv.time.sel(time=[time_grib]).values)})

        T = xr.DataArray(ds_tuv['t'].sel(time=[time_grib]).values, dims=['time', 'hybrid', 'latitude', 'longitude'])
            
        u_regridded = u_regridder(ds_tuv['u'].sel(time=[time_grib]))     
        U = xr.DataArray(u_regridded.values, dims=['time', 'hybrid', 'latitude', 'longitude_1']) 
        
        v_regridded = v_regridder(ds_tuv['v'].sel(time=[time_grib]))
        V = xr.DataArray(v_regridded.values, dims=['time', 'hybrid', 'latitude_1', 'longitude'])
        
        LNSP_3d = ds_lnsp.lnsp.sel(time=[time_grib])
        LNSP_4d = LNSP_3d.expand_dims(dim={'hybrid_1':1})
        LNSP = xr.DataArray(LNSP_4d.values, dims=('time', 'hybrid_1', 'latitude', 'longitude'))
        
        lnsp_regridded1 = u_regridder(LNSP_4d)
        LNSP_1 = xr.DataArray(lnsp_regridded1.values, dims=['time', 'hybrid_1', 'latitude', 'longitude_1'])  
        
        lnsp_regridded2 = v_regridder(LNSP_4d)
        LNSP_2 = xr.DataArray(lnsp_regridded2.values, dims=['time', 'hybrid_1', 'latitude_1', 'longitude'])                       
                     
        ds_out['hybrid'] = hybrid
        ds_out['hybrid_1'] = hybrid_1
        ds_out['latitude'] = latitude
        ds_out['latitude_1'] = latitude_1
        ds_out['longitude'] = longitude
        ds_out['longitude_1'] = longitude_1
        ds_out['T'] = T
        ds_out['U'] = U
        ds_out['V'] = V
        ds_out['LNSP'] = LNSP
        ds_out['LNSP_1'] = LNSP_1
        ds_out['LNSP_2'] = LNSP_2
        
        ds_out.time.values[0] = time_out

        ds_out.to_netcdf(path="/g/data/p66/rj9627/ERA5/processed/era5_1deg_ml137_"+date_list[t].strftime('%Y')+date_list[t].strftime('%m')+date_list[t].strftime('%d')+"{:02d}".format(t_hr[hr])+".nc", mode="w", format="NETCDF3_CLASSIC", unlimited_dims='time')
        print(f'Processed file for date: {date_list[t]}, time:{t_hr[hr]}')
    
exit()
