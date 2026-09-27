# ************************************************************************************
# Functions used to preprocess and compare ACCESS-EMS-GBR model output to observations
# ************************************************************************************

# resample_obs        : Resample observations to match model output frequency
# align_times         : Align time stamps of observations and model output
# get_model4coord     : Get model output for a given lat/lon coordinate
# get_model4shiptrack : Get model output for lat/lon coordinates along a ship track
# calc_air_density    : Calculate the density of air in cm-3
# convert_aer_number_in_mode  : Convert aerosol NMR from particles/mol to particles/cm3
# calc_N_gt_r         : Calculate aerosol number concentration w. dry radius > r
# calc_n10            : Calculate N10 (aerosol number concentration w. dry diameter > 10 nm)
# calc_ccn            : Calculate CCN for a given supersaturation
# construct_size_dist : Construct aerosol size distribution
# calc_aero_mass      : Calculate total aerosol masses
# calc_aod            : Calculate total AOD
# calc_winds          : Calculate wind speed and direction from U and V
# calc_bias           : Calculate model bias (absolute or percentage)
# calc_rmse           : Calculate model RMSE
# calc_nmbf           : Calculate normalised mean bias factor (Yu et al., 2006)
# calc_spearmanr      : Calculate Spearman's ranked correlation coefficient
# generate_summary_table : Generate a table of obs and model summary stats and model bias metrics


import xarray as xr
import numpy as np
import datetime as dt
import pandas as pd
from tabulate import tabulate
import scipy
from scipy import stats
import xesmf as xe
import math
from scipy.stats import rankdata, t


# *********************************************************************
# Resample observations to match model frequency
# *********************************************************************
# df    : pd.Dataframe w. observations 
# freq  : resample frequency ('h' for hourly)
# l_avg : True to calculate time mean over freq, False for instantaneous

def resample_obs(df, freq, l_avg):
    
    df = df.copy()
     
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
        df = df.set_index('time', drop=False)
    else:
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df = df.sort_index()
    
    if l_avg:
        numeric_cols = df.select_dtypes(include='number').columns
        other_cols = df.select_dtypes(exclude='number').columns
        df_numeric = df[numeric_cols].resample(freq).mean() # mean for numeric columns
        df_other = df[other_cols].resample(freq).first()    # first value for non-numeric columns
        df_out = pd.concat([df_numeric, df_other], axis=1)  # combine
        
    else:
        bin_label = df.index.floor(freq) # assign each timestamp to bin according to 'freq' (eg hourly, daily bins)
        df['bin'] = bin_label
        df = df.groupby('bin').apply(lambda x: x.ffill().bfill()) # Fill missing data with the next valid value within freq
        df = df.drop(columns='bin')
        df_out = df.resample(freq).asfreq()
        
    df_out['time'] = df_out.index
    return df_out


# *********************************************************************
# Align time stamps
# *********************************************************************
# model_ds : xr.Dataset w. model output
# obs_df   : pd.Dataframe w. observations
        
def align_times(model_ds, obs_df):

    model_ds = model_ds.copy()
    obs_df = obs_df.copy()

    # Ensure time coord is formatted correctly
    model_time = pd.to_datetime(model_ds.time.values)
    if 'time' in obs_df.columns:
        obs_df['time'] = pd.to_datetime(obs_df['time'], utc=True).dt.tz_localize(None)
        obs_df = obs_df.set_index('time')
    else:
        obs_df.index = pd.to_datetime(obs_df.index, utc=True).tz_localize(None)
    obs_df = obs_df.sort_index()
    obs_time = pd.to_datetime(obs_df.index)

    # Common timestamp range
    start = max(model_time.min(), obs_time.min())
    end = min(model_time.max(), obs_time.max())
    if start >= end:
        print('    WARNING align_times: no overlapping time range, returning None.')
        return None, None

    # Slice datasets to overlapping timestamps
    model_ds = model_ds.sel(time=slice(start, end))
    obs_df = obs_df.loc[start:end]
    model_time = pd.to_datetime(model_ds.time.values)
    obs_time = pd.to_datetime(obs_df.index)

    # Now check that timestamps are aligned and return 
    if len(model_time) != len(obs_time):
        print('    WARNING align_times: different number of timestamps, returning None.')
        print('Model first 5:', model_time[:5])
        print('Obs first 5:  ', obs_time[:5])
        return None, None

    if not np.array_equal(model_time, obs_time):
        print('    WARNING align_times: timestamps are not identical, returning None.')
        for i, (mt, ot) in enumerate(zip(model_time, obs_time)):
            if mt != ot:
                print('First mismatch at index', i)
                print('Model:', mt)
                print('Obs:  ', ot)
                break
        return None, None

    # Return aligned datasets
    obs_df['time'] = obs_df.index
    return model_ds, obs_df

            

# *********************************************************************
# Get model output for a given coordinate
# *********************************************************************
# ds      : xr.Dataset w. model output
# lat/lon : lat/lon coordinate to extract model output

def get_model4coord(ds, lat, lon):
    if ('latitude' in ds.dims) or ('latitude_0' in ds.dims): # UM regular grid - interpolate to coord
        out_ds = ds.interp(latitude=lat, latitude_0=lat, longitude=lon, longitude_0=lon, method='linear')
    elif ('j_centre' in ds.dims): # eReefs curvilinear grid - get closest coord
        abslat = np.abs(ds.y_centre - lat)
        abslon = np.abs(ds.x_centre - lon)
        c = np.maximum(abslon, abslat)
        ([yloc],[xloc]) = np.where(c == np.min(c))
        out_ds = ds.isel(j_centre=yloc, i_centre=xloc)
    return out_ds


# *********************************************************************
# Get model output along ship track
# *********************************************************************
# ds        : xr.Dataset w. model output
# ship_lats : ship lats (array)
# ship_lons : ship lons (array)
# ship_time : ship timestamps (array)

def get_model4shiptrack(ds, ship_lats, ship_lons, ship_time):
    if np.all(ds.time.values == ship_time):
        for idx,t in enumerate(ds.time.values):
            ds_tmp = get_model4coord(ds.sel(time=t), ship_lats[idx], ship_lons[idx])
            if idx == 0:
                ds_out = ds_tmp
            else:
                ds_out = xr.combine_nested([ds_out, ds_tmp], 'time')
        return ds_out
    else:
        print('    get_model4shiptrack failed - check timestamps.')
        return ds


# *********************************************************************
# Constants for aerosol calculations
# *********************************************************************
Rd = 287.05               # specific gas constant
cp = 1005.46              # J/kg/K
zboltz = 1.3807e-23       # Boltzmans constant
avo = 6.022e23            # Avogadros number
mm_da = avo * zboltz / Rd # molecular mass of dry air (kg/mol)


# *********************************************************************
# Calculate air density in cm-3
# *********************************************************************
# air_pres : air pressure (Pa) (m01s00i408)
# air_temp : air temperature (K) (m01s00i004)
# Assumes inputs are xr.DataArrays, outputs xr.DataArray

def calc_air_density(air_pres, air_temp):
    air_dens = air_pres / (air_temp * zboltz * 1.0e6)
    air_dens = air_dens.assign_attrs({'description':'air number density', 'units':'cm-3'})
    return air_dens


# *********************************************************************
# Convert aerosol number densities from particles/mol to particles/cm3
# *********************************************************************
# aero_ds   : xr.dataset containing aerosol number densities (particles/mol)
# air_dens  : xr.DataArray of air density (from calc_air_density)
# Returns xr.dataset with all aerosol mode number densities converted to cm-3

def convert_aer_number_in_mode(aero_ds, air_dens):
    aer_num = ['m01s34i101','m01s34i103','m01s34i107','m01s34i113','m01s34i119'] # soluble nucleation, Aitken, accumulation, coarse, insol Aitken  
    for imode in aer_num:
        if aero_ds[imode].attrs.get('units') != 'cm-3':
            aero_ds[imode] = aero_ds[imode] * air_dens
            aero_ds[imode] = aero_ds[imode].assign_attrs({'units':'cm-3'})
    return aero_ds


# *********************************************************************
# Calculate aerosol number concentration with dry radius > r
# *********************************************************************
# aero_ds   : xr.Dataset containing aerosol number densities
# cutoff_r  : minimum dry radius in m
# Uses lognormal_cumulative_to_r to integrate number < r for each mode, where:
    # N = aerosol number density in mode
    # r = cutoff radius in m
    # rbar = mode mean dry radius
    # sigma = mode geometric standard deviation
# Returns xr.DataArray

def lognormal_cumulative_to_r(N,r,rbar,sigma):
    total_to_r=(N/2.0)*(1.0+scipy.special.erf(np.log(r/rbar)/np.sqrt(2.0)/np.log(sigma)))
    return total_to_r

def calc_N_gt_r(aero_ds, cutoff_r):
    sigma_g = [1.59, 1.59, 1.4, 2.0, 1.59, 1.59, 2.0]
    nd = xr.concat([aero_ds['m01s34i101'], aero_ds['m01s34i103'], aero_ds['m01s34i107'], aero_ds['m01s34i113'], aero_ds['m01s34i119']], "mode")
    rbardry = xr.concat([aero_ds['m01s38i401'], aero_ds['m01s38i402'], aero_ds['m01s38i403'], aero_ds['m01s38i404'], aero_ds['m01s38i405']], "mode")
    rbardry = rbardry / 2 # convert from diameter -> radius
    nd = nd.to_numpy()
    rbardry = rbardry.to_numpy()
    
    # loop over number of modes
    nmodes = nd.shape[0]
    for imode in range(nmodes):
        nd_lt_r_this_mode = lognormal_cumulative_to_r(nd[imode,:], cutoff_r, rbardry[imode,:], sigma_g[imode])
        nd_gt_r_this_mode = nd[imode] - nd_lt_r_this_mode
        if (imode == 0):
            nd_gt_r = nd_gt_r_this_mode
        else:
            nd_gt_r = nd_gt_r + nd_gt_r_this_mode
    nd_gt_r = xr.DataArray(nd_gt_r, dims=list(aero_ds['m01s34i101'].dims), attrs={'description':'Calculated aerosol number concentration','units':'cm-3'})
    return nd_gt_r


# *********************************************************************
# Calculate N10
# *********************************************************************
# First convert aerosol mode number densities from particles/mol –> cm-3,
# then calculate N10 using calc_N_gt_r function above

# aero_ds = xr.dataset containing aerosol mode number densities and diameters
# Returns xr.DataArray

def calc_n10(aero_ds):
    cutoff_r = 5 *1e-9 # 5 nm radius -> m
    N10 = calc_N_gt_r(aero_ds, cutoff_r)
    return N10        


# *********************************************************************
# Calculate activation dry diameter at a given SS
# *********************************************************************

# Adapted from S. Fiddes: https://github.com/sfiddes/ACCESS_aerosol_eval/blob/main/process_aer_along_ship.py 

# ss    : supersaturation (%) (returns scalar)

def calc_kohler_easy_way(ss):
    rh = 1.0+(ss/100.)
    temp = 298.64
    
    # Calculate A factor (p787, S&P1998)
    A = 0.66/temp # in microns
    lnrh = np.log(rh)
    
    # Calculate B factor (p788, S&P1998) [at critical droplet diameter]
    B = (4.0*A**3.)/(27.0*lnrh**2.) # in microns^
    
    # solute mass (g particle-1) [at critical droplet diameter]
    ms = B*98.0/(3.0*3.44e13) # in g (3.44e13 all soluble mass per particle # of dissociating ions in mols/m) 
    
    # convert to kg particle-1
    ms = ms/1000.0 # in kg
    
    # calculate particle dry volume (assume particle density of 1800 kg m-3)
    vol = ms/1800. # in m3
    act_r = ((3.0*vol)/(4.0*np.pi))**(1.0/3.0) # in m
    
    return(2*act_r*1e9)


# *********************************************************************
# Calculate CCN at a given supersaturation
# *********************************************************************
# aero_ds   : xr.Dataset containing aerosol number concentration & diameter of each mode
# ss        : supersaturation (%)
# Returns xr.DataArray
# Uses calc_N_gt_r on soluble modes only.

def calc_ccn(aero_ds, ss):
    
    nd = xr.concat([aero_ds['m01s34i101'], aero_ds['m01s34i103'], aero_ds['m01s34i107'], aero_ds['m01s34i113']], "mode")
    rbardry = xr.concat([aero_ds['m01s38i401'], aero_ds['m01s38i402'], aero_ds['m01s38i403'], aero_ds['m01s38i404']], "mode")
    rbardry = rbardry / 2 # convert from diameter -> radius
    
    sigma_g = [1.59, 1.59, 1.4, 2.0, 1.59, 1.59, 2.0]
    nmodes = nd.shape[0]
    cutoff_d = calc_kohler_easy_way(ss)
    cutoff_r = (cutoff_d/2) *1e-9
    print(f'    Activation diameter at {ss}% SS:', cutoff_d, 'nm')
          
    # loop over number of modes
    for imode in range(nmodes):
        nd_lt_r_this_mode = lognormal_cumulative_to_r(nd[imode], cutoff_r, rbardry[imode], sigma_g[imode])
        nd_gt_r_this_mode = nd[imode] - nd_lt_r_this_mode
        if (imode == 0):
            nd_gt_r = nd_gt_r_this_mode
        else:
            nd_gt_r = nd_gt_r + nd_gt_r_this_mode

    nd_gt_r = xr.DataArray(nd_gt_r, name=f'CCN_{ss}', attrs={'units':'cm-3'})
    
    return nd_gt_r


# *********************************************************************
# Construct aerosol size distribution
# *********************************************************************

# Adapted from S. Fiddes: https://github.com/sfiddes/ACCESS_aerosol_eval/blob/main/process_aer_along_ship.py  

# aero_ds   : xr.Dataset containing aerosol number concentration & diameter of each mode
# bins      : number of bins to interpolate model output onto, or an array of specified bin sizes (set l_setbins to True)
# Optional  :
        # l_setbins   : True if 'bins' is an array of set bin diameters, False is 'bins' is the number of bins to derive
        # rmin / rmax : smallest / largest radius (m) to derive bin diameters if l_setbins is False
        
# Returns an xr.Dataset containing variables 'diameter' (bin diameters) and 'sizedist' (number concentration in each bin)

def construct_size_dist(ds, bins, **kwargs):

    if 'l_setbins' in kwargs:
        l_setbins = kwargs['l_setbins']
        nbins = len(bins)
    else:
        l_setbins = False
        nbins = bins
        rmin = kwargs['rmin']
        rmax = kwargs['rmax']
        
    nsteps = len(ds.time)
    nmodes = 5 # Using 5 mode setup

    # Get arrays of mode number concentration & diameter
    nd = xr.concat([ds['m01s34i101'], ds['m01s34i103'], ds['m01s34i107'], ds['m01s34i113'], ds['m01s34i119']], "mode") # Number conc. (mode x time ...)
    rbardry = xr.concat([ds['m01s38i401'], ds['m01s38i402'], ds['m01s38i403'], ds['m01s38i404'], ds['m01s38i405']], "mode") # Mode diameter (mode x time ...)
    rbardry = rbardry / 2 # convert from diameter -> radius
    nd = nd.to_numpy()
    rbardry = rbardry.to_numpy()
    dnd = np.zeros((nbins,nsteps))

    # Calculate size distribution
    for i in range(nsteps):
        if l_setbins:
            dndlogd,dryr_mid = calculate_size_dist(nmodes, nd, rbardry, i, bins, l_setbins=l_setbins)
        else:
            dndlogd,dryr_mid = calculate_size_dist(nmodes, nd, rbardry, i, bins, l_setbins=l_setbins, rmin=rmin, rmax=rmax)
        dnd[:,i] = dndlogd[nmodes,:]

    # Format data
    dryr_mid = dryr_mid *2. *1.0e9 # Convert radius (in m) to diameter (in nm)
    sizedist = xr.DataArray(dnd,coords=[dryr_mid,ds.time], dims=['diameter','time']).to_dataset(name='sizedist')
    ds_out = sizedist    
    ds_out['sizedist'] = ds_out['sizedist'].assign_attrs({'units':'dN / dlogD (cm-3)'})
    ds_out['diameter'] = ds_out['diameter'].assign_attrs({'units':'nm'})
    
    return ds_out


# *********************************************************************
# Evaluate dndlogd for each size bin
# *********************************************************************
# nmodes  : # of aerosol size modes
# nd      : np.array of aerosol number concentration (mode x time, ...)
# rbardry : np.array of mode mean dry diameters in m (mode x time, ...)
# t       : time index
# bins    : bins from construct_size_dist

def calculate_size_dist(nmodes, nd, rbardry, t, bins, **kwargs):

    if 'l_setbins' in kwargs:
        l_setbins = kwargs['l_setbins']
        nbins = len(bins)
    else:
        l_setbins = False
        nbins = bins
        rmin = kwargs['rmin']
        rmax = kwargs['rmax']
        
    sigma_g = [1.59,1.59,1.4,2.0,1.59,1.59,2.0]
    
    # Determine which modes are active
    mode = np.zeros((nmodes), dtype=bool)
    for imode in range(nmodes):
        mode[imode] = np.isfinite(nd[imode,:]).any()
    
    # Define points for calculating size distribution
    if l_setbins:
        dryr_mid = (bins / 2)   # use specified bin middle radius in m  
    else:
        dryr_mid = np.zeros(nbins) # derive bin middle radius in m
        dryr_int = np.zeros(nbins+1)
        for ipt in range (nbins+1):
            logr = np.log(rmin)+(np.log(rmax)-np.log(rmin))*np.float(ipt)/np.float(nbins)
            dryr_int[ipt] = np.exp(logr)
        for ipt in range (nbins):
            dryr_mid[ipt] = 10.0**(0.5*(np.log10(dryr_int[ipt+1])+np.log10(dryr_int[ipt]))) 
            
    dndlogd = np.zeros((nmodes+1,nbins)) # number of modes, plus total number    
        
    for ipt in range(nbins):
        for imode in range(nmodes):  
            if (mode[imode]):
                dndlogd[imode,ipt] = lognormal_dndlogd(nd[imode,t],
                                                       dryr_mid[ipt]*2,
                                                       rbardry[imode,t]*2,
                                                       sigma_g[imode])
            else:
                dndlogd[imode,ipt] = np.nan
        dndlogd[nmodes,ipt] = np.sum(dndlogd[0:nmodes,ipt])
        
    return dndlogd, dryr_mid


# *********************************************************************
# Calculate lognormal distribution (dn/dlogd) at diameter d
# *********************************************************************
# nd      : np.array of aerosol number concentration (mode x time, ...)
# d       : required dry diameter (m) 
# dbar    : mode mean dry diameter (m)
# sigma_g : mode geometric standard deviation 

def lognormal_dndlogd(nd, d, dbar, sigma_g):

    xpi = 3.14159265358979323846e0

    numexp = -(np.log(d)-np.log(dbar))**2.0
    denomexp = 2.0*np.log(sigma_g)*np.log(sigma_g)

    denom = np.sqrt(2.0*xpi)*np.log(sigma_g)

    dndlnd = (nd/denom)*np.exp(numexp/denomexp)

    dndlogd = 2.303*dndlnd

    return dndlogd


# *********************************************************************
# Calculate total aerosol masses (ug/m3)
# *********************************************************************
# aero_ds       : xr.Dataset containing aerosol MMRs (in kg/kg)
# air_density   : xr.DataArray of air density (cm3)
# spec          : aerosol species (so4, ss, oc, bc)
# Return xr.DataArray of total spec mass across aerosol modes

def calc_aero_mass(aero_ds, air_density, spec):
    
    if spec == 'so4':
        stash = ['m01s34i102','m01s34i104','m01s34i108','m01s34i114']
    elif spec == 'ss':
        if 'm01s34i127' in list(aero_ds.keys()):
            stash = ['m01s34i111','m01s34i117','m01s34i127'] # include Aitken SS
        else:
            stash = ['m01s34i111','m01s34i117']
    elif spec == 'oc':
        stash = ['m01s34i126','m01s34i106','m01s34i121','m01s34i110','m01s34i116']
    elif spec == 'bc':
        stash = ['m01s34i105','m01s34i109','m01s34i115','m01s34i120']
        
    # First convert from kg/kg –> ug/m3
    conversion_factor = air_density * mm_da / avo * 1e6 * 1e9   # convert to ug/m3
    aero_ds_tmp = aero_ds.copy()
    for istash in stash:
        if aero_ds_tmp[istash].attrs.get('units') != 'ug m-3':
            aero_ds_tmp[istash] = aero_ds_tmp[istash] * conversion_factor
    
    # Now sum over modes
    total_mass = sum(aero_ds_tmp[istash] for istash in stash)
    total_mass = total_mass.assign_attrs({'units':'ug m-3'})

    return total_mass


# *********************************************************************
# Calculate total AOD
# *********************************************************************
# ds : xr.Dataset containing AOD for modes
# Returns xr.DataArray of total AOD

def calc_aod(ds):
    aod = xr.DataArray(ds['m01s02i300'] + ds['m01s02i301'] + ds['m01s02i302'] + ds['m01s02i303'], # sol Ait, accum, coarse, insol Ait
                       dims=list(ds.m01s02i300.dims), attrs={'description':'Total AOD', 'units':''}) 
    return aod


# *********************************************************************
# Calculate wind speed and direction
# *********************************************************************
# u : m01s00i002
# v : m01s00i003
# Returns 2x xr.DataArrays of wind speed (m/s) and wind direction (deg)

def calc_winds(u,v):
    
    # Regrid U and V wind components to standard model grid
    u_grid = xr.Dataset({'latitude':(['latitude'], u.latitude.values), 'longitude':(['longitude'], u.longitude_0.values)})
    v_grid = xr.Dataset({'latitude':(['latitude'], v.latitude_0.values), 'longitude':(['longitude'], v.longitude.values)})
    tgt_grid = xr.Dataset({'latitude':(['latitude'], u.latitude.values), 'longitude':(['longitude'], v.longitude.values)})

    u_regridder = xe.Regridder(u_grid, tgt_grid, method='bilinear', periodic=True)
    u_regridded = u_regridder(u)

    v_regridder = xe.Regridder(v_grid, tgt_grid, method='bilinear', periodic=True)
    v_regridded = v_regridder(v)
        
    # Calculate wind speed
    WS = (u_regridded**2 + v_regridded**2) ** 0.5
    WS_da = xr.DataArray(WS, name='WindSpeed', dims=list(u_regridded.dims), attrs={'description':'Wind speed (level_height_0)', 'units':'m s-1'})
        
    # Calculate wind direction
    WD = (270-np.arctan2(v_regridded,u_regridded)*180/math.pi)%360
    WD_da = xr.DataArray(WD, name='WindDir', dims=list(u_regridded.dims), attrs={'description':'Wind direction (level_height_0)', 'units':'degree'})
    
    return WS_da, WD_da


# *********************************************************************
# Calculate model bias metrics
# *********************************************************************

# obs   : observations (np.array)
# model : model (np.array)
# optional
        # l_perc : True to calculate percentage bias, False for absolute bias

# Model bias
def calc_bias(obs, model, **kwargs):
    if 'l_perc' in kwargs:
        l_perc = kwargs['l_perc']
    else:
        l_perc = False
    if l_perc:
        diff = ((model - obs) / obs)*100
        bias = np.nanmean(diff)
    else:
        diff = model - obs
        bias = np.nanmean(diff)
    return bias 

# RMSE
def calc_rmse(obs, model):
    rmse = np.sqrt(np.mean((model - obs)**2))
    return rmse

# Normalised Mean Bias Factor (NMBF)
# Reference: Yu et al. (2006). New unbiased symmetric metrics for evaluation of air quality models, Atmospheric Science Letters, 7, 26-34. https://doi.org/10.1002/asl.125
def calc_nmbf(obs, model):
    if np.shape(model)[0] == np.shape(obs)[0]:
        if np.nanmean(model) >= np.nanmean(obs):
            F = np.nansum(model-obs) / np.nansum(obs)
        else:
            F = np.nansum(model-obs) / np.nansum(model)
        return F
    else:
        print('    Warning: calc_nmbf failed - check timestamps.')
        return None
    
# Spearmans correlation coefficient
def calc_spearmanr(obs, model):
    n = len(model)
    if n < 3:
        return np.nan, np.nan
    rx = rankdata(model)
    ry = rankdata(obs)
    rho = np.corrcoef(rx, ry)[0, 1]
    rho = np.clip(rho, -0.9999999, 0.9999999) # prevent floating point rounding
    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
    pval = 2 * t.sf(np.abs(t_stat), df=n-2)
    return rho, pval


# *********************************************************************
# Generate a table of summary stats and model bias metrics
# *********************************************************************

# Prints a table and returns a pd.DataFrame that can be saved to .csv

# comparison_dict : dictionary containing {variable : (obs, [model_runs]} definitions for comparisons
# variable is the var name (str)
# obs is an array of observations
# model_runs can point to multiple np.arrays for different model runs in a list eg. 'rev1_atmos_N10', 'rev2_atmos_N10' etc

def generate_summary_table(comparison_dict):
    
    r_prec = 4 # rounding precision
    headers = ['Variable','Source','Mean','Min','Max','SD','Bias','Bias (%)','RMSE','NMBF','Spearmans r','p']
    table = []
    
    for ivar, (obs, model_runs) in comparison_dict.items():
        
        # Exclude any NaNs
        is_valid = pd.notna(obs) & pd.notna(model_runs[-1])
        obs = obs[is_valid]
        
        # Observation stats
        obs_n = len(obs)
        obs_avg = round(obs.mean(), r_prec)
        obs_min = round(obs.min(), r_prec)
        obs_max = round(obs.max(), r_prec)
        obs_sd = round(obs.std(), r_prec)
        table.append([ivar, f'Observed (n={obs_n})', obs_avg, obs_min, obs_max, obs_sd, '','','','','',''])
        
        # Model stats and biases
        for irun in model_runs:
            irun = irun[is_valid]
            irun_avg = round(irun.mean(), r_prec)
            irun_min = round(irun.min(), r_prec)
            irun_max = round(irun.max(), r_prec)
            irun_sd = round(irun.std(), r_prec)
            irun_bias = round(calc_bias(obs, irun), r_prec)
            irun_biasp = round(calc_bias(obs, irun, l_perc=True), r_prec)
            irun_rmse = round(calc_rmse(obs, irun), r_prec)
            irun_nmbf = round(calc_nmbf(obs, irun), r_prec)
            irun_r, irun_p = calc_spearmanr(obs, irun)
            irun_r = round(irun_r, r_prec)
            if irun_p < 0.001:
                irun_p = '<0.001'
            else:
                irun_p = np.round(irun_p, r_prec)
            run_id = irun.name.split('_')[0]
            table.append(['', f'Modelled ({run_id})', irun_avg, irun_min, irun_max, irun_sd, irun_bias, irun_biasp, irun_rmse, irun_nmbf, irun_r, irun_p])
            
    #print(tabulate(table, headers=headers, tablefmt='fancy_grid', stralign='left', numalign='left'))
    df = pd.DataFrame(table, columns=headers)
    df['Variable'] = df['Variable'].replace('', np.nan).ffill()
    return df
    
