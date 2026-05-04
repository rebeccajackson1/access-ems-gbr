'''
Functions used to format and compare ACCESS-EMS-GBR model output to observations
________________________________________________________________________________

resample_obs         : Resmaple observations to match model output frequency.
align_times          : Align time stamps of observations and model output.
get_model4coord      : Get model output for a given lat/lon coordinate.
get_model4shiptrack  : Get model output for lat/lon coordinates along a ship track.
calc_bias            : Calculate model bias (actual or %).
calc_rmse            : Calculate model RMSE.
calc_nmbf            : Calculate model Normalised Mean Bias Factor.
calc_spearmanr       : Calculate Spearman's correlation coefficient (stats.spearmanr)
calc_spearmanr_fast  : Calculate Spearman's correlation coefficient (stats.rankdata).
generate_summary_table : Display a table of summary and bias metrics.
'''

import xarray as xr
import numpy as np
import datetime as dt
import pandas as pd
from tabulate import tabulate
from scipy import stats


# --------------------------------------------------------------------------------------------------
# Resample observations to match model frequency
# --------------------------------------------------------------------------------------------------

# df    : pd.Dataframe w. observations 
# freq  : resample frequency ('h' for hourly)
# l_avg : True to calculate time mean over freq, False for instantaneous

def resample_obs(df, freq, l_avg):
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time', drop=False)
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if l_avg:
        df_out = df.resample(freq).mean()
    else:
        bin_label = df.index.floor(freq) # assign each timestamp to bin according to 'freq' (eg hourly, daily bins)
        df['bin'] = bin_label
        df = df.groupby('bin').apply(lambda x: x.ffill().bfill()) # Fill missing data with the next valid value within freq
        df = df.drop(columns='bin')
        df_out = df.resample(freq).asfreq()
    df_out['time'] = df_out.index
    return df_out


# --------------------------------------------------------------------------------------------------
# Align time stamps
# --------------------------------------------------------------------------------------------------

# model_ds : xr.Dataset w. model output
# obs_df   : pd.Dataframe w. observations

def align_times(model_ds, obs_df):
    model_ds = model_ds.copy()
    obs_df = obs_df.copy()
    
    # Ensure time coord is formatted correctly
    model_time = pd.to_datetime(model_ds.time.values)
    if 'time' in obs_df.columns:
        obs_df['time'] = pd.to_datetime(obs_df['time'])
        obs_df = obs_df.set_index('time')
    else:
        obs_df.index = pd.to_datetime(obs_df.index)
    obs_time = obs_df.index

    # Common timestamp range
    start = max(model_time[0], obs_time[0])
    end = min(model_time[-1], obs_time[-1])
    if start >= end:
        print('Warning: align times failed - no overlapping time range between model and observations.')
    else:
        model_ds = model_ds.sel(time=slice(start, end))
        obs_df = obs_df.loc[start:end]
        model_time = pd.to_datetime(model_ds.time.values)
        obs_time = pd.to_datetime(obs_df.index)
        if len(model_time) == len(obs_time) and np.array_equal(model_time, obs_time):
            print('Timestamps aligned.')
            obs_df['time'] = obs_df.index
            return model_ds, obs_df
        else:
            print('Warning: align_times failed - check timestamps.')


# --------------------------------------------------------------------------------------------------
# Get model output for a given coordinate
# --------------------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------------------
# Get model output along ship track
# --------------------------------------------------------------------------------------------------

# ds  : xr.Dataset w. model output
# gps : pd.Dataframe w. ship coords

def get_model4shiptrack(ds, gps):
    if np.all(ds.time.values == gps['time'].values):
        ship_lat = gps['lat.interp'].values
        ship_lon = gps['lon.interp'].values
        for idx,t in enumerate(ds.time.values):
            ds_tmp = get_model4coord(ds.sel(time=t), ship_lat[idx], ship_lon[idx])
            if idx == 0:
                ds_out = ds_tmp
            else:
                ds_out = xr.combine_nested([ds_out, ds_tmp], 'time')
        return ds_out
    else:
        print('get_model4shiptrack failed - check timestamps.')
        return ds


# --------------------------------------------------------------------------------------------------
# Model bias
# --------------------------------------------------------------------------------------------------

# obs_ar   : np.array of obs values
# model_ar : np.array of corresponding model values
# optional args
#    l_perc : True for percentage bias, False for absolute

def calc_bias(model_ar, obs_ar, **kwargs):
    if 'l_perc' in kwargs:
        l_perc = kwargs['l_perc']
    else:
        l_perc = False
    if l_perc:
        diff = ((model_ar - obs_ar) / obs_ar)*100
        bias = np.nanmean(diff)
    else:
        diff = model_ar - obs_ar
        bias = np.nanmean(diff)
    return bias 


# --------------------------------------------------------------------------------------------------
# RMSE
# --------------------------------------------------------------------------------------------------

# obs_ar   :   np.array of obs values
# model_ar :   np.array of model values

def calc_rmse(model_ar, obs_ar):
    rmse = np.sqrt(np.mean((model_ar - obs_ar)**2))
    return rmse


# --------------------------------------------------------------------------------------------------
# Normalised Mean Bias Factor (NMBF)
# --------------------------------------------------------------------------------------------------

# Reference: Yu et al. (2006). New unbiased symmetric metrics for evaluation of air quality models, Atmospheric Science Letters, 7, 26-34. https://doi.org/10.1002/asl.125
# obs_ar   : np.array of obs values (no NaNs)
# model_ar : np.array of corresponding model values

def calc_nmbf(model_ar, obs_ar):
    if np.shape(model_ar)[0] == np.shape(obs_ar)[0]:
        if np.nanmean(model_ar) >= np.nanmean(obs_ar):
            F = np.nansum(model_ar-obs_ar) / np.nansum(obs_ar)
        else:
            F = np.nansum(model_ar-obs_ar) / np.nansum(model_ar)
        return F
    else:
        print('Warning: calc_nmbf failed - check timestamps.')


# --------------------------------------------------------------------------------------------------
# Spearman's correlation coefficient (obs v model)
# --------------------------------------------------------------------------------------------------
# obs_ar   : np.array of obs values
# model_ar : np.array of corresponding modelled values

def calc_spearmanr(model_ar, obs_ar):
    result = stats.spearmanr(obs_ar, model_ar, nan_policy='omit', alternative='two-sided')
    return result

def calc_spearmanr_fast(model_ar, obs_ar):
    from scipy.stats import rankdata, t
    n = len(model_ar)
    if n < 3:
        return np.nan, np.nan
    rx = rankdata(model_ar)
    ry = rankdata(obs_ar)
    rho = np.corrcoef(rx, ry)[0, 1]
    rho = np.clip(rho, -0.9999999, 0.9999999) # prevent floating point rounding
    t_stat = rho * np.sqrt((n - 2) / (1 - rho**2))
    pval = 2 * t.sf(np.abs(t_stat), df=n-2)
    return rho, pval


# --------------------------------------------------------------------------------------------------
# Summary stats and bias metrics
# --------------------------------------------------------------------------------------------------

# Displays a table with summary and bias metrics for vars at each site/campaign

# obs_df     : pd.Dataframe w. obs values
# model_ds   : xr.Dataset w. model output
# obs_vars   : list of variable names in obs_df
# model_vars : list of variable names in model_ds
# Optional args
#    model_ds2 : Additional xr.Dataset w. model output (for revised aerosol run)

def generate_summary_table(obs_df, model_ds, obs_vars, model_vars, **kwargs):

    r_prec = 4 # rounding precision
    obs_ds = xr.Dataset.from_dataframe(obs_df)
    obs_ds_aligned, model_ds_aligned = xr.align(obs_ds, model_ds, join='inner')

    if 'model_ds2' in kwargs:
        model_ds2 = kwargs['model_ds2']
        obs_ds2 = xr.Dataset.from_dataframe(obs_df)
        obs_ds_aligned2, model_ds_aligned2 = xr.align(obs_ds2, model_ds2, join='inner')
        
    headers = ['Variable','# points','Observed\nmean','Observed\nmin','Observed\nmax','Observed\nSD',
               'Model\nmean','Model\nmin','Model\nmax','Model\nSD','Bias','Bias (%)','RMSE','NMBF','Spearmans r','p']
    
    table = []
    for v,ivar in enumerate(model_vars):

        obs_da = obs_ds_aligned[obs_vars[v]].copy()
        model_da = model_ds_aligned[ivar]
        valid = ~(np.isnan(obs_da) | np.isnan(model_da))
        obs_arr = obs_da.to_numpy()
        model_arr = model_da.to_numpy()
        obs_arr = obs_arr[valid]
        model_arr = model_arr[valid]
        nobs = len(obs_arr)

        # Stats
        obs_mean = obs_arr.mean()
        obs_min = obs_arr.min()
        obs_max = obs_arr.max()
        obs_sd = obs_arr.std(ddof=1)
                
        model_mean = model_arr.mean()
        model_min = model_arr.min()
        model_max = model_arr.max()
        model_sd = model_arr.std(ddof=1)

        BIAS = calc_bias(model_arr, obs_arr)
        pBIAS = calc_bias(model_arr, obs_arr, l_perc=True)
        RMSE = calc_rmse(model_arr, obs_arr)
        NMBF = calc_nmbf(model_arr, obs_arr)
        correlation,pvalue = calc_spearmanr_fast(model_arr, obs_arr)
        if pvalue<0.001:
            p = '<0.001'
        else:
            p = np.round(pvalue,r_prec)
                
        table.append([f'{obs_vars[v]} ({ivar})', nobs,
                      np.round(obs_mean,r_prec), np.round(obs_min,r_prec), np.round(obs_max,r_prec), np.round(obs_sd,r_prec),
                      np.round(model_mean,r_prec), np.round(model_min,r_prec), np.round(model_max,r_prec), np.round(model_sd,r_prec),
                      np.round(BIAS,r_prec), np.round(pBIAS,r_prec), np.round(RMSE,r_prec), np.round(NMBF,r_prec), np.round(correlation,r_prec), p])
    
        if 'model_ds2' in kwargs:
            obs_da2 = obs_ds_aligned2[obs_vars[v]].copy()
            model_da2 = model_ds_aligned2[ivar]
            valid2 = ~(np.isnan(obs_da2) | np.isnan(model_da2))
            obs_arr2 = obs_da2.to_numpy()
            model_arr2 = model_da2.to_numpy()
            obs_arr2 = obs_arr2[valid2]
            model_arr2 = model_arr2[valid2]
            nobs2 = len(obs_arr2)
            
            # Stats 
            model_mean2 = model_arr2.mean()
            model_min2 = model_arr2.min()
            model_max2 = model_arr2.max()
            model_sd2 = model_arr2.std(ddof=1)

            BIAS2 = calc_bias(model_arr2, obs_arr2)
            pBIAS2 = calc_bias(model_arr2, obs_arr2, l_perc=True)
            RMSE2 = calc_rmse(model_arr2, obs_arr2)
            NMBF2 = calc_nmbf(model_arr2, obs_arr2)
            correlation2,pvalue2 = calc_spearmanr_fast(model_arr2, obs_arr2)
            if pvalue2<0.001:
                p2 = '<0.001'
            else:
                p2 = np.round(pvalue2,r_prec)
                    
            table.append([f'{obs_vars[v]} ({ivar}) revised', nobs2, '', '', '', '',
                          np.round(model_mean2,r_prec), np.round(model_min2,r_prec), np.round(model_max2,r_prec), np.round(model_sd2,r_prec),
                          np.round(BIAS2,r_prec), np.round(pBIAS2,r_prec), np.round(RMSE2,r_prec), np.round(NMBF2,r_prec), np.round(correlation2,r_prec), p2])
            
    print(tabulate(table, headers=headers, tablefmt='fancy_grid', stralign='left', numalign='left'))        

