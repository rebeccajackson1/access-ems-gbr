'''
Format ACCESS-EMS-GBR model output and observations for model evaluation
_________________________________________________________________________

1. Reads in observations and model output
2. Formats observations (time stamps and data filtering)
3. Resamples observations to hourly means for consistency with the model output
4. Gets model output for observed lat, lon and level (slow step)
5. Calculates required model variables (N10, CCN, aerosol masses, size distribution etc)
6. Aligns time stamps and combines into a single pd.DataFrame for each dataset
7. Saves the processed combined datasets to .csv
'''

import xarray as xr
from xarray.coding.times import CFTimedeltaCoder
import numpy as np
import datetime as dt
import pandas as pd
import math

from functions import *


# *********************************************************************
# Inputs
# *********************************************************************

obs_dir = '/home/578/rj9627/python/shared/access-ems-gbr-evaluation/obs/'
model_dir = '/g/data/p66/rj9627/UM_NS/output/RRAP/ACCESS-EMS-GBR/'
out_dir = '/g/data/p66/rj9627/UM_NS/output/RRAP/obs_comparison/'

suite_ctl = 'ds142'  # Control aerosol setup
suite_rev = 'dn070'  # Revised aerosol setup
rgn = 'NE_Aus'
res = '4km'
fcst = 48
lev_height = 100 # max height to read (m)
aod_level = 2    # AOD wavelength (550 nm)

# Model run ID, label, suite and path to output
model_runs = {'control' : ['Control', suite_ctl, model_dir+'u-'+suite_ctl+'/um2nc/'],                                # Control aerosol setup
              'rev1'    : ['Ait SS + EDGAR', suite_rev, model_dir+'u-'+suite_rev+'/um2nc_AitSS_EDGAR/u-'],           # +Ait SS, EDGAR anthropogenic emissions
              'rev2'    : ['Ait SS + EDGAR + BLN', suite_rev, model_dir+'u-'+suite_rev+'/um2nc_AitSS_EDGAR_BLN/u-'], # rev1 + BLN + terrestrial monoterpenes
              'rev3'    : ['Revised', suite_rev, model_dir+'u-'+suite_rev+'/um2nc/']}                                # rev2 + SOA scaling + PMOC size adjustments

# Observation datasets           
obs_datasets = {# RRAP Feb 2022 ship data (N10, CCN) 
                'RRAP22_gps'  : obs_dir+'RRAP_Feb22/Consolidated_Magnetic_GPS_Position_seconds.csv',
                'RRAP22_N10'  : obs_dir+'RRAP_Feb22/Diagnostics.rds.csv',
                'RRAP22_CCN'  : obs_dir+'RRAP_Feb22/sd.aero.delim.csv',
                
                # RRAP Mar 2023 Heron Island Research Station (N10, CCN, size dist, masses, meteorology)
                'RRAP23_N10' : obs_dir+'RRAP_Mar23/CPC.corrected.txt',                                
                'RRAP23_CCN' : obs_dir+'RRAP_Mar23/CCN.corrected.txt', 
                'RRAP23_SD'  : obs_dir+'RRAP_Mar23/SEMS.APS.combined.corrected.csv', 
                'RRAP23_AMS' : obs_dir+'RRAP_Mar23/AMS.HR.csv',
                'RRAP23_BC'  : obs_dir+'RRAP_Mar23/BC.corrected.csv', 
                'RRAP23_met' : obs_dir+'RRAP_Mar23/Met.corrected.txt.xlsx',
                
                # AERONET Lucinda (AOD)
                'AERONET22'   : obs_dir+'AERONET/20220131_20220302_Lucinda.lev20',                    
                'AERONET23'   : obs_dir+'AERONET/20230228_20230402_Lucinda.lev20',
                
                # AIMS weather station - Davies Reef
                'AIMS22_davies_temp' : obs_dir+'AIMS_WS/Davies_Reef_Air_Temperature_202202-202203.csv',   
                'AIMS22_davies_pres' : obs_dir+'AIMS_WS/Davies_Reef_Air_Pressure_202202-202203.csv',
                'AIMS22_davies_wspd' : obs_dir+'AIMS_WS/Davies_Reef_Wind_Speed_(Scalar_avg_10_min)_202202-202203.csv',
                'AIMS22_davies_wdir' : obs_dir+'AIMS_WS/Davies_Reef_Wind_Direction_(Scalar_Average_10_Minutes)_202202-202203.csv',
                'AIMS23_davies_temp' : obs_dir+'AIMS_WS/Davies_Reef_Air_Temperature_202303-202304.csv',
                'AIMS23_davies_pres' : obs_dir+'AIMS_WS/Davies_Reef_Air_Pressure_202303-202304.csv',
                'AIMS23_davies_wspd' : obs_dir+'AIMS_WS/Davies_Reef_Wind_Speed_(Scalar_avg_10_min)_202303-202304.csv',
                'AIMS23_davies_wdir' : obs_dir+'AIMS_WS/Davies_Reef_Wind_Direction_(Scalar_Average_10_Minutes)_202303-202304.csv',
                
                # AIMS weather station - Heron Island (2022 only)
                'AIMS22_heron_temp' : obs_dir+'AIMS_WS/Heron_Island_Air_Temperature_202202-202203.csv',   
                'AIMS22_heron_pres' : obs_dir+'AIMS_WS/Heron_Island_Air_Pressure_202202-202203.csv',
                'AIMS22_heron_wspd' : obs_dir+'AIMS_WS/Heron_Island_Wind_Speed_(Scalar_avg_30_min)_202202-202203.csv',
                'AIMS22_heron_wdir' : obs_dir+'AIMS_WS/Heron_Island_Wind_Direction_(Scalar_Average_30_Minutes)_202202-202203.csv',
                
                # EcoRRAP seawater loggers - Davies Reef (lagoon flat, front shallow and front deep sites)
                'EcoRRAP22_davies_LF_temp' : obs_dir+'EcoRRAP_Seawater/Davies_Reef_Lagoon_Flat_RBRSolo3-Temp-logger_202202-202202.csv',
                'EcoRRAP22_davies_FS_temp' : obs_dir+'EcoRRAP_Seawater/Davies_Reef_Front_Shallow_RBRSolo3-Temp-logger_202202-202202.csv',
                'EcoRRAP22_davies_FD_CTD'  : obs_dir+'EcoRRAP_Seawater/Davies_Reef_Front_Deep_SBE-CTD_202202-202202.csv',
                'EcoRRAP22_davies_FD_PAR'  : obs_dir+'EcoRRAP_Seawater/Davies_Reef_Front_Deep_RBRSolo3-Licor-sensor_202202-202202.csv',
                'EcoRRAP23_davies_LF_temp' : obs_dir+'EcoRRAP_Seawater/Davies_Reef_Lagoon_Flat_RBRSolo3-Temp-logger_202303-202303.csv',
                'EcoRRAP23_davies_FS_temp' : obs_dir+'EcoRRAP_Seawater/Davies_Reef_Front_Shallow_RBRSolo3-Temp-logger_202303-202303.csv',
                'EcoRRAP23_davies_FD_CTD'  : obs_dir+'EcoRRAP_Seawater/Davies_Reef_Front_Deep_SBE-CTD_202303-202303.csv',
                'EcoRRAP23_davies_FD_PAR' : obs_dir+'EcoRRAP_Seawater/Davies_Reef_Front_Deep_RBRSolo3-Licor-sensor_202303-202303.csv',
                
                # EcoRRAP seawater loggers - Heron Island (lagoon shallow, front shallow and front deep sites) - N/A for 2022 FD and 2023 LS 
                'EcoRRAP22_heron_LS_temp' : obs_dir+'EcoRRAP_Seawater/Heron_Island_Lagoon_Shallow_RBRSolo3-Temp-logger_202202-202202.csv',
                'EcoRRAP22_heron_FS_temp' : obs_dir+'EcoRRAP_Seawater/Heron_Island_Front_Shallow_RBRSolo3-Temp-logger_202202-202202.csv',
                'EcoRRAP22_heron_FD_PAR'  : obs_dir+'EcoRRAP_Seawater/Heron_Island_Front_Deep_RBRSolo3-Licor-sensor_202202-202202.csv',
                'EcoRRAP23_heron_FS_temp' : obs_dir+'EcoRRAP_Seawater/Heron_Island_Front_Shallow_RBRSolo3-Temp-logger_202303-202303.csv',
                'EcoRRAP23_heron_FD_CTD'  : obs_dir+'EcoRRAP_Seawater/Heron_Island_Front_Deep_SBE-CTD_202303-202303.csv',
                'EcoRRAP23_heron_FD_PAR'  : obs_dir+'EcoRRAP_Seawater/Heron_Island_Front_Deep_RBRSolo3-Licor-sensor_202303-202303.csv'}


# *********************************************************************
# Read in model output
# *********************************************************************

print('Reading model output')

model_dsets = {}
for irun,(ilabel,isuite,ipath) in model_runs.items():
    
    # Atmosphere (NE Aus) - read in model output for the control, rev1, rev2 and rev3 runs
    run_id = irun+'_atmos'
    model_dsets[run_id] = {}
    model_dsets[run_id]['2022'] = xr.open_mfdataset(ipath+isuite+'-'+rgn+'-'+res+'-'+str(fcst)+'hrFcst-2022*', concat_dim='time', combine='nested', data_vars='different', decode_timedelta=CFTimedeltaCoder(decode_via_units=True), chunks={})
    model_dsets[run_id]['2023'] = xr.open_mfdataset(ipath+isuite+'-'+rgn+'-'+res+'-'+str(fcst)+'hrFcst-2023*', concat_dim='time', combine='nested', data_vars='different', decode_timedelta=CFTimedeltaCoder(decode_via_units=True), chunks={})
    
    # Format datasets (simpler indexing and interpolation)
    for iyr in model_dsets[run_id].keys():
        model_dsets[run_id][iyr]['m01s00i002'] = model_dsets[run_id][iyr]['m01s00i002'].swap_dims({'model_level_number':'level_height_0'})  # U wind
        model_dsets[run_id][iyr]['m01s00i003'] = model_dsets[run_id][iyr]['m01s00i003'].swap_dims({'model_level_number':'level_height_0'})  # V wind
        model_dsets[run_id][iyr] = model_dsets[run_id][iyr].swap_dims({'model_level_number':'level_height'}) # all other variables
        model_dsets[run_id][iyr] = model_dsets[run_id][iyr].sel(level_height=slice(0,lev_height), level_height_0=slice(0,lev_height), pseudo_level=aod_level) # subset levels
    
    print(f'    Read {run_id}')
    
    # Ocean (GBR4) - read in BGC output for the revised run only
    if irun == 'rev3':
        run_id = irun+'_ocean'
        model_dsets[run_id] = {}
        model_dsets[run_id]['2022'] = xr.open_mfdataset(model_dir+'u-'+suite_rev+'/gbr4_bgc/gbr4_bgc_all_hr_2022*', concat_dim='record', combine='nested', data_vars='different')
        model_dsets[run_id]['2023'] = xr.open_mfdataset(model_dir+'u-'+suite_rev+'/gbr4_bgc/gbr4_bgc_all_hr_2023*', concat_dim='record', combine='nested', data_vars='different')
        
        # Format dataset coords
        for iyr in model_dsets[run_id].keys():
            model_dsets[run_id][iyr] = model_dsets[run_id][iyr].swap_dims({'k_centre':'z_centre', 'record':'t'})
            model_dsets[run_id][iyr] = model_dsets[run_id][iyr].assign_coords(t = pd.to_datetime(model_dsets[run_id][iyr]['t'], utc=True).tz_localize(None).round('h'))
            model_dsets[run_id][iyr] = model_dsets[run_id][iyr].rename({'t':'time'})
    
        print(f'    Read {run_id}')
            
        
# *********************************************************************
# Read in observations
# *********************************************************************

print('Reading observations')

obs_dsets = {}

for iobs,ipath in obs_datasets.items():
    if iobs == 'RRAP23_met':
        obs_dsets[iobs] = pd.read_excel(ipath)
    else:                        
        if iobs in ['RRAP23_N10','RRAP23_CCN','RRAP23_SD','RRAP23_AMS','RRAP23_BC']:
            sep = ';'
            header = 0
        elif 'AERONET' in iobs:
            sep = ','
            header = 6
        else:
            sep = ','
            header = 0
        obs_dsets[iobs] = pd.read_csv(ipath, sep=sep, header=header)
    
    print(f'    Read {iobs}')
    

# *********************************************************************
# Format observation timestamps
# *********************************************************************

print('Formatting observation timestamps')

# First, manually combine 'date' and 'time' for some obs
obs_dsets['RRAP23_BC']['DATETIME'] = obs_dsets['RRAP23_BC']['Date(yyyy/MM/dd);'] + obs_dsets['RRAP23_BC']['Time(hh:mm:ss);']

obs_dsets['AERONET22']['DATE'] = pd.to_datetime(obs_dsets['AERONET22']['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
obs_dsets['AERONET22']['TIME'] = pd.to_timedelta(obs_dsets['AERONET22']['Time(hh:mm:ss)'])
obs_dsets['AERONET22']['DATETIME'] = obs_dsets['AERONET22']['DATE'] + obs_dsets['AERONET22']['TIME']

obs_dsets['AERONET23']['DATE'] = pd.to_datetime(obs_dsets['AERONET23']['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
obs_dsets['AERONET23']['TIME'] = pd.to_timedelta(obs_dsets['AERONET23']['Time(hh:mm:ss)'])
obs_dsets['AERONET23']['DATETIME'] = obs_dsets['AERONET23']['DATE'] + obs_dsets['AERONET23']['TIME']

# Now ensure dates are datetime and labelled 'time'
for iobs, df in obs_dsets.items():   
    if 'time' not in df.columns:
        for date_label in ['timestamp','date','DATETIME','Date(yyyy/MM/dd);','Date(dd:mm:yyyy)','time_utc']:
            if date_label in df.columns:
                df = df.rename(columns={date_label : 'time'})
                break
            
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        df = df.set_index(pd.DatetimeIndex(df['time']), drop=False)
        df = df[~df.index.duplicated(keep='first')] # Remove any duplicate timestamps (RRAP22_N10 and RRAP23_BC)        

    obs_dsets[iobs] = df


# *********************************************************************
# Filter RRAP 2022 ship N10 and CCN data
# *********************************************************************

print('Filtering RRAP ship N10 and CCN data')

# Remove N10 data flagged as MCB or fogging activity 
flags = ['Fogging_Salt','Fogging_Fresh','Fogging_Generator','MCB_Salt','MCB_Fresh','MCB_Generator','MCB_M1','MCB_M2','MCB_M3']
obs_dsets['RRAP22_N10'] = obs_dsets['RRAP22_N10'].loc[~obs_dsets['RRAP22_N10']['Activity'].isin(flags)]

# Remove data where N10 >= 10,000 cm-3 (flagged as potential ship influence) and < 1 (invalid)
obs_dsets['RRAP22_N10']['CN_guard_cloud'] = obs_dsets['RRAP22_N10']['CN_guard_cloud'].where((obs_dsets['RRAP22_N10']['CN_guard_cloud'] >= 1) & (obs_dsets['RRAP22_N10']['CN_guard_cloud'] < 10000), np.nan)
obs_dsets['RRAP22_N10']['CN_mag'] = obs_dsets['RRAP22_N10']['CN_mag'].where((obs_dsets['RRAP22_N10']['CN_mag'] >= 1) & (obs_dsets['RRAP22_N10']['CN_mag'] < 10000), np.nan)

# Read in ship meteorology for N10 and CCN filtering
met_guardian = pd.read_csv(obs_dir+'RRAP_Feb22/Met.corrected.1.txt', sep=';', low_memory=False) 
met_magnetic = pd.read_csv(obs_dir+'RRAP_Feb22/Met.corrected.2.txt', sep=';', low_memory=False)
met_combined = pd.concat([met_guardian, met_magnetic])
met_combined = met_combined.rename(columns={'DATETIME':'time'})
met_combined['time'] = pd.to_datetime(met_combined['time'])
met_combined = met_combined.set_index(pd.DatetimeIndex(met_combined['time']), drop=False)
met_combined = met_combined[~met_combined.index.duplicated(keep='first')]

# Filter RRAP 2022 ship N10 data by winds to exclude ship influence

# Resample to 1 min means
met_tmp = met_combined.copy()
met_tmp = resample_obs(met_tmp, '1min', l_avg=True) 
obs_dsets['RRAP22_N10'] = resample_obs(obs_dsets['RRAP22_N10'], '1min', l_avg=True)

# Align timestamps
start = max(obs_dsets['RRAP22_N10'].index.min(), met_tmp.index.min())
end = min(obs_dsets['RRAP22_N10'].index.max(), met_tmp.index.max())
aligned_dates = pd.date_range(start, end, freq='1min')
met_tmp = met_tmp.reindex(aligned_dates)
obs_dsets['RRAP22_N10'] = obs_dsets['RRAP22_N10'].reindex(aligned_dates)

# Filter data by winds
if obs_dsets['RRAP22_N10'].index.equals(met_tmp.index):
    obs_dsets['RRAP22_N10']['CN_guard_cloud'] = np.where(((met_tmp['DIR'] >= 270) | (met_tmp['DIR'] <= 90)) & (met_tmp['SPEED'] > 2), obs_dsets['RRAP22_N10']['CN_guard_cloud'], np.nan)
    obs_dsets['RRAP22_N10']['CN_mag'] = np.where(((met_tmp['DIR'] >= 270) | (met_tmp['DIR'] <= 90)) & (met_tmp['SPEED'] > 2), obs_dsets['RRAP22_N10']['CN_mag'], np.nan)
    print('    Filtered ship N10 data.')
else:
    print('    Filtering ship N10 data failed due to timestamp mismatch.')
    
# Now combine ship N10
obs_dsets['RRAP22_N10']['CN_combined'] = obs_dsets['RRAP22_N10']['CN_guard_cloud'].fillna(obs_dsets['RRAP22_N10']['CN_mag'])


# Filter RRAP 2022 ship CCN data by winds to exclude ship influence

# Resample to 1 min means
met_tmp = met_combined.copy()
met_tmp = resample_obs(met_tmp, '1min', l_avg=True) 
obs_dsets['RRAP22_CCN'] = resample_obs(obs_dsets['RRAP22_CCN'], '1min', l_avg=True)

# Align timestamps
start = max(obs_dsets['RRAP22_CCN'].index.min(), met_tmp.index.min())
end = min(obs_dsets['RRAP22_CCN'].index.max(), met_tmp.index.max())
aligned_dates = pd.date_range(start, end, freq='1min')
met_tmp = met_tmp.reindex(aligned_dates)
obs_dsets['RRAP22_CCN'] = obs_dsets['RRAP22_CCN'].reindex(aligned_dates)

# Filter data by winds
if obs_dsets['RRAP22_CCN'].index.equals(met_tmp.index):
    obs_dsets['RRAP22_CCN']['conc'] = np.where(((met_tmp['DIR'] >= 270) | (met_tmp['DIR'] <= 90)) & (met_tmp['SPEED'] > 2), obs_dsets['RRAP22_CCN']['conc'], np.nan)
    print('    Filtered ship CCN data.')
else:
    print('    Filtering ship CCN data failed due to timestamp mismatch.')


# *********************************************************************
# Resample observations to hourly means for consistency with model
# *********************************************************************

print('Resampling observations to hourly means')

# First separate CCN into SS bins
for i_ss in [0.1, 0.2, 0.3, 0.5, 0.7]:
    obs_dsets['RRAP22_CCN'][f'CCN_{i_ss}'] = obs_dsets['RRAP22_CCN']['conc'].where(obs_dsets['RRAP22_CCN']['SS'] == i_ss)
    obs_dsets['RRAP23_CCN'][f'CCN_{i_ss}'] = obs_dsets['RRAP23_CCN']['conc'].where(obs_dsets['RRAP23_CCN']['SS'] == i_ss)
 
# Now resample
for iobs, df in obs_dsets.items():
    obs_dsets[iobs] = resample_obs(df, 'h', l_avg=True)


# *********************************************************************
# Combine observations for common locations and time periods
# *********************************************************************

print('Combining observation datasets for common locations and time periods')

# RRAP 2023 aerosol masses
combine_obs = ['RRAP23_AMS','RRAP23_BC']
df_combined = obs_dsets[combine_obs[0]].copy()
for df_name in combine_obs[1:]:
    df_combined = df_combined.join(obs_dsets[df_name], how='outer', lsuffix='', rsuffix=f'_{df_name}') # use 'outer' to align by index (time), while keeping all timestamps from both datasets.
obs_dsets['RRAP23_aero_masses'] = df_combined


# AIMS weather station - Davies Reef
combine_obs = ['AIMS22_davies_temp','AIMS22_davies_pres','AIMS22_davies_wspd','AIMS22_davies_wdir']
df_combined = obs_dsets[combine_obs[0]].copy()
for df_name in combine_obs[1:]:
    df = obs_dsets[df_name]
    df_combined = df_combined.join(df, how='outer', lsuffix='', rsuffix=f'_{df_name}') # use 'outer' to align by index (time), while keeping all timestamps from both datasets.
obs_dsets['AIMS22_davies_combined'] = df_combined
    
combine_obs = ['AIMS23_davies_temp','AIMS23_davies_pres','AIMS23_davies_wspd','AIMS23_davies_wdir']
df_combined = obs_dsets[combine_obs[0]].copy()
for df_name in combine_obs[1:]:
    df = obs_dsets[df_name]
    df_combined = df_combined.join(df, how='outer', lsuffix='', rsuffix=f'_{df_name}')
obs_dsets['AIMS23_davies_combined'] = df_combined


# AIMS weather station - Heron Island (2022 only)
combine_obs = ['AIMS22_heron_temp','AIMS22_heron_pres','AIMS22_heron_wspd','AIMS22_heron_wdir']
df_combined = obs_dsets[combine_obs[0]].copy()
for df_name in combine_obs[1:]:
    df = obs_dsets[df_name]
    df_combined = df_combined.join(df, how='outer', lsuffix='', rsuffix=f'_{df_name}')
obs_dsets['AIMS22_heron_combined'] = df_combined


# EcoRRAP seawater loggers - Davies Reef
obs_dsets['EcoRRAP22_davies_LF_temp'] = obs_dsets['EcoRRAP22_davies_LF_temp'].rename(columns={'TEMP_degrees_Celsius':'LF_TEMP_degrees_Celsius', 'lat':'LF_lat', 'lon':'LF_lon'})
obs_dsets['EcoRRAP22_davies_FS_temp'] = obs_dsets['EcoRRAP22_davies_FS_temp'].rename(columns={'TEMP_degrees_Celsius':'FS_TEMP_degrees_Celsius', 'lat':'FS_lat', 'lon':'FS_lon'})
obs_dsets['EcoRRAP22_davies_FD_CTD'] = obs_dsets['EcoRRAP22_davies_FD_CTD'].rename(columns={'TEMP_degrees_Celsius':'FD_TEMP_degrees_Celsius', 'PSAL_1':'FD_PSAL_1', 'lat':'FD_CTD_lat', 'lon':'FD_CTD_lon'})
obs_dsets['EcoRRAP22_davies_FD_PAR'] = obs_dsets['EcoRRAP22_davies_FD_PAR'].rename(columns={'PAR_umole_m-2_s-1':'FD_PAR_umole_m-2_s-1', 'lat':'FD_PAR_lat', 'lon':'FD_PAR_lon'})

obs_dsets['EcoRRAP23_davies_LF_temp'] = obs_dsets['EcoRRAP23_davies_LF_temp'].rename(columns={'TEMP_degrees_Celsius':'LF_TEMP_degrees_Celsius', 'lat':'LF_lat', 'lon':'LF_lon'})
obs_dsets['EcoRRAP23_davies_FS_temp'] = obs_dsets['EcoRRAP23_davies_FS_temp'].rename(columns={'TEMP_degrees_Celsius':'FS_TEMP_degrees_Celsius', 'lat':'FS_lat', 'lon':'FS_lon'})
obs_dsets['EcoRRAP23_davies_FD_CTD'] = obs_dsets['EcoRRAP23_davies_FD_CTD'].rename(columns={'TEMP_degrees_Celsius':'FD_TEMP_degrees_Celsius', 'PSAL_1':'FD_PSAL_1', 'lat':'FD_CTD_lat', 'lon':'FD_CTD_lon'})
obs_dsets['EcoRRAP23_davies_FD_PAR'] = obs_dsets['EcoRRAP23_davies_FD_PAR'].rename(columns={'PAR_umole_m-2_s-1':'FD_PAR_umole_m-2_s-1', 'lat':'FD_PAR_lat', 'lon':'FD_PAR_lon'})

combine_obs = ['EcoRRAP22_davies_LF_temp','EcoRRAP22_davies_FS_temp','EcoRRAP22_davies_FD_CTD','EcoRRAP22_davies_FD_PAR']
df_combined = obs_dsets[combine_obs[0]].copy()
for df_name in combine_obs[1:]:
    df = obs_dsets[df_name]
    df_combined = df_combined.join(df, how='outer', lsuffix='', rsuffix=f'_{df_name}')
obs_dsets['EcoRRAP22_davies_combined'] = df_combined

combine_obs = ['EcoRRAP23_davies_LF_temp','EcoRRAP23_davies_FS_temp','EcoRRAP23_davies_FD_CTD','EcoRRAP23_davies_FD_PAR']
df_combined = obs_dsets[combine_obs[0]].copy()
for df_name in combine_obs[1:]:
    df = obs_dsets[df_name]
    df_combined = df_combined.join(df, how='outer', lsuffix='', rsuffix=f'_{df_name}')
obs_dsets['EcoRRAP23_davies_combined'] = df_combined


# EcoRRAP seawater loggers - Heron Island
obs_dsets['EcoRRAP22_heron_LS_temp'] = obs_dsets['EcoRRAP22_heron_LS_temp'].rename(columns={'TEMP_degrees_Celsius':'LS_TEMP_degrees_Celsius', 'lat':'LS_lat', 'lon':'LS_lon'})
obs_dsets['EcoRRAP22_heron_FS_temp'] = obs_dsets['EcoRRAP22_heron_FS_temp'].rename(columns={'TEMP_degrees_Celsius':'FS_TEMP_degrees_Celsius', 'lat':'FS_lat', 'lon':'FS_lon'})
obs_dsets['EcoRRAP22_heron_FD_PAR'] = obs_dsets['EcoRRAP22_heron_FD_PAR'].rename(columns={'PAR_umole_m-2_s-1':'FD_PAR_umole_m-2_s-1', 'lat':'FD_PAR_lat', 'lon':'FD_PAR_lon'})

obs_dsets['EcoRRAP23_heron_FS_temp'] = obs_dsets['EcoRRAP23_heron_FS_temp'].rename(columns={'TEMP_degrees_Celsius':'FS_TEMP_degrees_Celsius', 'lat':'FS_lat', 'lon':'FS_lon'})
obs_dsets['EcoRRAP23_heron_FD_CTD'] = obs_dsets['EcoRRAP23_heron_FD_CTD'].rename(columns={'TEMP_degrees_Celsius':'FD_TEMP_degrees_Celsius', 'PSAL_1':'FD_PSAL_1', 'lat':'FD_CTD_lat', 'lon':'FD_CTD_lon'})
obs_dsets['EcoRRAP23_heron_FD_PAR'] = obs_dsets['EcoRRAP23_heron_FD_PAR'].rename(columns={'PAR_umole_m-2_s-1':'FD_PAR_umole_m-2_s-1', 'lat':'FD_PAR_lat', 'lon':'FD_PAR_lon'})

combine_obs = ['EcoRRAP22_heron_LS_temp','EcoRRAP22_heron_FS_temp','EcoRRAP22_heron_FD_PAR']
df_combined = obs_dsets[combine_obs[0]].copy()
for df_name in combine_obs[1:]:
    df = obs_dsets[df_name]
    df_combined = df_combined.join(df, how='outer', lsuffix='', rsuffix=f'_{df_name}')
obs_dsets['EcoRRAP22_heron_combined'] = df_combined

combine_obs = ['EcoRRAP23_heron_FS_temp','EcoRRAP23_heron_FD_CTD','EcoRRAP23_heron_FD_PAR']
df_combined = obs_dsets[combine_obs[0]].copy()
for df_name in combine_obs[1:]:
    df = obs_dsets[df_name]
    df_combined = df_combined.join(df, how='outer', lsuffix='', rsuffix=f'_{df_name}') 
obs_dsets['EcoRRAP23_heron_combined'] = df_combined


# *********************************************************************
# Format observed met and seawater variables
# *********************************************************************

# Convert wind speed from km/hr –> m/s
obs_dsets['AIMS22_davies_combined']['Wind_Speed_(Scalar_avg_10_min)'] = obs_dsets['AIMS22_davies_combined']['Wind_Speed_(Scalar_avg_10_min)'] *5/18
obs_dsets['AIMS23_davies_combined']['Wind_Speed_(Scalar_avg_10_min)'] = obs_dsets['AIMS23_davies_combined']['Wind_Speed_(Scalar_avg_10_min)'] *5/18
obs_dsets['AIMS22_heron_combined']['Wind_Speed_(Scalar_avg_30_min)'] = obs_dsets['AIMS22_heron_combined']['Wind_Speed_(Scalar_avg_30_min)'] *5/18

# Convert seawater PAR from umol/m2/s –> mol/m2/hr
obs_dsets['EcoRRAP22_davies_combined']['FD_PAR_umole_m-2_s-1'] = obs_dsets['EcoRRAP22_davies_combined']['FD_PAR_umole_m-2_s-1'] *1e-6 *60*60
obs_dsets['EcoRRAP23_davies_combined']['FD_PAR_umole_m-2_s-1'] = obs_dsets['EcoRRAP23_davies_combined']['FD_PAR_umole_m-2_s-1'] *1e-6 *60*60
obs_dsets['EcoRRAP22_heron_combined']['FD_PAR_umole_m-2_s-1'] = obs_dsets['EcoRRAP22_heron_combined']['FD_PAR_umole_m-2_s-1'] *1e-6 *60*60
obs_dsets['EcoRRAP23_heron_combined']['FD_PAR_umole_m-2_s-1'] = obs_dsets['EcoRRAP23_heron_combined']['FD_PAR_umole_m-2_s-1'] *1e-6 *60*60


# *********************************************************************
# Get model output for observed lat/lon/level
# *********************************************************************

print('Getting model output for observed lat/lon/level')

model_processed = {}

# RRAP 2022 ship track
print('    RRAP 2022 ship track')
obs_df = obs_dsets['RRAP22_gps']
for irun,dsets in model_dsets.items():
    if 'atmos' in irun:
        model_ds = dsets['2022']
        model_aligned, obs_aligned = align_times(model_ds, obs_df) # First align timestamps
        lat = obs_aligned['lat.interp'].values
        lon = obs_aligned['lon.interp'].values
        time = obs_aligned['time'].values
        level = 10 # m
        ds_out = get_model4shiptrack(model_aligned, lat, lon, time) # Now get model for shiptrack coords
        ds_out = ds_out.interp(level_height=level, level_height_0=level) # and level
        model_processed[irun] = {}
        model_processed[irun]['2022_rrap_ship'] = ds_out


# RRAP 2023 Heron Island Research Station
print('    RRAP 2023 Heron Island')
lat = -23.4426
lon = 151.9130
level = 5
for irun,dsets in model_dsets.items():
    if 'atmos' in irun:
        ds = dsets['2023']
        ds_out = get_model4coord(ds, lat, lon)
        ds_out = ds_out.interp(level_height=level, level_height_0=level)
        model_processed[irun]['2023_rrap_heron'] = ds_out


# AERONET Lucinda
print('    AERONET Lucinda')
lat = -18.5198
lon = 146.3861
level = 8
for irun,dsets in model_dsets.items():
    
    if 'atmos' in irun:
        ds = dsets['2022']
        ds_out = get_model4coord(ds, lat, lon)
        ds_out = ds_out.interp(level_height=level, level_height_0=level)
        model_processed[irun]['2022_aeronet'] = ds_out
        
        ds = dsets['2023']
        ds_out = get_model4coord(ds, lat, lon)
        ds_out = ds_out.interp(level_height=level, level_height_0=level)
        model_processed[irun]['2023_aeronet'] = ds_out


# AIMS weather station - Davies Reef
print('    AIMS weather station - Davies Reef')
lat = -18.8316
lon = 147.6345
level = 10
irun = 'rev3_atmos'
dsets = model_dsets[irun]
    
ds = dsets['2022']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(level_height=level, level_height_0=level)
model_processed[irun]['2022_aims_davies'] = ds_out
    
ds = dsets['2023']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(level_height=level, level_height_0=level)
model_processed[irun]['2023_aims_davies'] = ds_out
    

# AIMS weather station - Heron Island (2023 uses RRAP dataset)
print('    AIMS weather station - Heron Island')
lat = -23.4482
lon = 151.9839
level = 6
irun = 'rev3_atmos'
dsets = model_dsets[irun]
ds = dsets['2022']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(level_height=level, level_height_0=level)
model_processed[irun]['2022_aims_heron'] = ds_out

lat = -23.4426
lon = 151.9130
level = 5
ds = dsets['2023']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(level_height=level, level_height_0=level)
model_processed[irun]['2023_met_heron'] = ds_out


# EcoRRAP seawater loggers - Davies Reef
print('    EcoRRAP - Davies Reef')
irun = 'rev3_ocean'
dsets = model_dsets[irun]
model_processed[irun] = {}
model_processed[irun]['2022_ecorrap_davies'] = {}
model_processed[irun]['2023_ecorrap_davies'] = {}

# Lagoon flat
lat = obs_dsets['EcoRRAP22_davies_combined'].LF_lat.iloc[0]
lon = obs_dsets['EcoRRAP22_davies_combined'].LF_lon.iloc[0]
level = -0.5
ds = dsets['2022']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2022_ecorrap_davies']['Davies_LF'] = ds_out

lat = obs_dsets['EcoRRAP23_davies_combined'].LF_lat.iloc[0]
lon = obs_dsets['EcoRRAP23_davies_combined'].LF_lon.iloc[0]
level = -0.5
ds = dsets['2023']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2023_ecorrap_davies']['Davies_LF'] = ds_out

# Front shallow
lat = obs_dsets['EcoRRAP22_davies_combined'].FS_lat.iloc[0]
lon = obs_dsets['EcoRRAP22_davies_combined'].FS_lon.iloc[0]
level = -3.8
ds = dsets['2022']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2022_ecorrap_davies']['Davies_FS'] = ds_out

lat = obs_dsets['EcoRRAP23_davies_combined'].FS_lat.iloc[0]
lon = obs_dsets['EcoRRAP23_davies_combined'].FS_lon.iloc[0]
level = -3.8
ds = dsets['2023']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2023_ecorrap_davies']['Davies_FS'] = ds_out

# Front deep
lat = obs_dsets['EcoRRAP22_davies_combined'].FD_CTD_lat.iloc[0]
lon = obs_dsets['EcoRRAP22_davies_combined'].FD_CTD_lon.iloc[0]
level = -13
ds = dsets['2022']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2022_ecorrap_davies']['Davies_FD'] = ds_out

lat = obs_dsets['EcoRRAP23_davies_combined'].FD_CTD_lat.iloc[0]
lon = obs_dsets['EcoRRAP23_davies_combined'].FD_CTD_lon.iloc[0]
level = -13
ds = dsets['2023']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2023_ecorrap_davies']['Davies_FD'] = ds_out


# EcoRRAP seawater loggers - Heron Island
print('    EcoRRAP - Heron Island')
model_processed[irun]['2022_ecorrap_heron'] = {}
model_processed[irun]['2023_ecorrap_heron'] = {}

# Lagoon shallow (2022 only)
lat = obs_dsets['EcoRRAP22_heron_combined'].LS_lat.iloc[0]
lon = obs_dsets['EcoRRAP22_heron_combined'].LS_lon.iloc[0]
level = -5.0
ds = dsets['2022']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2022_ecorrap_heron']['Heron_LS'] = ds_out

# Front shallow
lat = obs_dsets['EcoRRAP22_heron_combined'].FS_lat.iloc[0]
lon = obs_dsets['EcoRRAP22_heron_combined'].FS_lon.iloc[0]
level = -5.0
ds = dsets['2022']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2022_ecorrap_heron']['Heron_FS'] = ds_out

lat = obs_dsets['EcoRRAP23_heron_combined'].FS_lat.iloc[0]
lon = obs_dsets['EcoRRAP23_heron_combined'].FS_lon.iloc[0]
level = -5.0
ds = dsets['2023']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2023_ecorrap_heron']['Heron_FS'] = ds_out

# Front deep (PAR only in 2022)
lat = obs_dsets['EcoRRAP22_heron_combined'].FD_PAR_lat.iloc[0]
lon = obs_dsets['EcoRRAP22_heron_combined'].FD_PAR_lon.iloc[0]
level = -14.5
ds = dsets['2022']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2022_ecorrap_heron']['Heron_FD'] = ds_out

lat = obs_dsets['EcoRRAP23_heron_combined'].FD_CTD_lat.iloc[0]
lon = obs_dsets['EcoRRAP23_heron_combined'].FD_CTD_lon.iloc[0]
level = -14.5
ds = dsets['2023']
ds_out = get_model4coord(ds, lat, lon)
ds_out = ds_out.interp(z_centre=level)
model_processed[irun]['2023_ecorrap_heron']['Heron_FD'] = ds_out


# *********************************************************************
# Calculate model variables 
# *********************************************************************

print('Calculating model variables')

mm_so4 = 96.06
mm_h2so4 = 98.079
so4_factor = mm_so4 / mm_h2so4 # ~0.9794146
oc_factor = 1.4

for irun in model_processed.keys():
    
    # Atmospheric datasets
    if 'atmos' in irun:
        
        # N10, CCN - calculate for RRAP 2022 and 2023 campaigns (control, rev1, rev2 and rev3)
        for ds_name in ['2022_rrap_ship', '2023_rrap_heron']:
            ds = model_processed[irun][ds_name]
            air_temp = ds['m01s00i004']
            air_press = ds['m01s00i408']
            air_density = calc_air_density(air_press, air_temp)
            if irun not in ['control_atmos', 'rev3_atmos']:
                ds = convert_aer_number_in_mode(ds, air_density) # convert aerosol NMR to cm-3 (already done at runtime for control and rev3)
            ds['N10'] = calc_n10(ds)                         # calculate N10
            for i_ss in [0.1, 0.2, 0.3, 0.5, 0.7]:           # calculate CCN at measured SS (%)
                ds['CCN_'+str(i_ss)] = calc_ccn(ds, i_ss)   
                
            # Filter model N10 as per ship observations: CN >= 10,000 cm-3 were considered 'polluted' (amongst other filters).
            if ds_name == '2022_rrap_ship':
                ds['N10'] = xr.where(ds['N10']<10000, ds['N10'], np.nan, keep_attrs=True)
           
            # Also calc aerosol masses for RRAP 2023 campaign (control, rev1, rev2 and rev3)
            if ds_name == '2023_rrap_heron':
                ds['aerosol_so4_mass'] = calc_aero_mass(ds, air_density, 'so4') *so4_factor # sulfate mass (convert model H2SO4 to SO4)
                ds['aerosol_org_mass'] = calc_aero_mass(ds, air_density, 'oc') *oc_factor   # organics mass (model assumes OM:OC ratio of 1.4)
                ds['aerosol_bc_mass'] = calc_aero_mass(ds, air_density, 'bc')               # black carbon mass
                
            # Put back into model_processed
            model_processed[irun][ds_name] = ds  
    
        # AERONET AOD
        for ds_name in ['2022_aeronet', '2023_aeronet']:
            ds = model_processed[irun][ds_name]
            ds['AOD'] = calc_aod(ds)
            model_processed[irun][ds_name] = ds  
        
        # Only need the following for the final revised run
        if 'rev3' in irun:
            
            # AIMS weather station meteorology (and Heron Island RS met for Mar 2023)
            for ds_name in ['2022_aims_davies', '2023_aims_davies', '2022_aims_heron', '2023_met_heron']:
                ds = model_processed[irun][ds_name]
                ds['m01s16i004'] = ds['m01s16i004'] - 273.15   # convert air temp from K –> C
                ds['m01s00i408'] = ds['m01s00i408'] /100       # convert air pressure from Pa –> hPa
                ds['wind_speed'] = (ds['m01s00i002']**2 + ds['m01s00i003']**2) ** 0.5                  # wind speed (m/s)
                ds['wind_dir'] = (270-np.arctan2(ds['m01s00i003'], ds['m01s00i002'])*180/math.pi)%360  # wind direction (deg N)             
                model_processed[irun][ds_name] = ds
        
    # Ocean datasets
    if 'rev3_ocean' in irun:
        for ds_name in ['2022_ecorrap_davies','2023_ecorrap_davies','2022_ecorrap_heron','2023_ecorrap_heron']:
            for iloc in model_processed[irun][ds_name].keys():
                ds = model_processed[irun][ds_name][iloc]
                ds['PAR'] = ds['PAR'] *60*60  # Convert PAR from mol/m2/s –> mol/m2/hr
                model_processed[irun][ds_name][iloc] = ds       


# *********************************************************************
# Align observation and model timestamps and put into pd.DataFrame
# *********************************************************************

print('Combining datasets')

# obs dataset name : ([model runs], model dataset name, [obs_vars], [model_vars], output file suffix)

comparison_dsets = {'RRAP22_N10' : (['control_atmos','rev1_atmos','rev2_atmos','rev3_atmos'], '2022_rrap_ship', ['CN_combined'], ['N10'], 'aerosol_N10'),
                    'RRAP22_CCN' : (['control_atmos','rev1_atmos','rev2_atmos','rev3_atmos'], '2022_rrap_ship', ['CCN_0.1','CCN_0.2','CCN_0.3','CCN_0.5','CCN_0.7'], ['CCN_0.1','CCN_0.2','CCN_0.3','CCN_0.5','CCN_0.7'], 'aerosol_CCN'),
                    'RRAP23_N10' : (['control_atmos','rev1_atmos','rev2_atmos','rev3_atmos'], '2023_rrap_heron', ['conc'], ['N10'], 'aerosol_N10'),
                    'RRAP23_CCN' : (['control_atmos','rev1_atmos','rev2_atmos','rev3_atmos'], '2023_rrap_heron', ['CCN_0.1','CCN_0.2','CCN_0.3','CCN_0.5','CCN_0.7'], ['CCN_0.1','CCN_0.2','CCN_0.3','CCN_0.5','CCN_0.7'], 'aerosol_CCN'),
                    'RRAP23_aero_masses' : (['control_atmos','rev1_atmos','rev2_atmos','rev3_atmos'], '2023_rrap_heron', ['HRSO4','HROrg','BC2;'], ['aerosol_so4_mass','aerosol_org_mass','aerosol_bc_mass'], 'aerosol_mass'),
                    'AERONET22'  : (['control_atmos','rev1_atmos','rev2_atmos','rev3_atmos'], '2022_aeronet', ['AOD_551nm'], ['AOD'], 'aerosol_AOD'),
                    'AERONET23'  : (['control_atmos','rev1_atmos','rev2_atmos','rev3_atmos'], '2023_aeronet', ['AOD_532nm'], ['AOD'], 'aerosol_AOD'),
                    'AIMS22_davies_combined' : (['rev3_atmos'], '2022_aims_davies',  ['Air_Temperature','Air_Pressure','Wind_Speed_(Scalar_avg_10_min)','Wind_Direction_(Scalar_Average_10_Minutes)'], ['m01s16i004','m01s00i408','wind_speed','wind_dir'], 'met'),
                    'AIMS23_davies_combined' : (['rev3_atmos'], '2023_aims_davies', ['Air_Temperature','Air_Pressure','Wind_Speed_(Scalar_avg_10_min)','Wind_Direction_(Scalar_Average_10_Minutes)'], ['m01s16i004','m01s00i408','wind_speed','wind_dir'], 'met'),
                    'AIMS22_heron_combined'  : (['rev3_atmos'], '2022_aims_heron', ['Air_Temperature','Air_Pressure','Wind_Speed_(Scalar_avg_30_min)','Wind_Direction_(Scalar_Average_30_Minutes)'], ['m01s16i004','m01s00i408','wind_speed','wind_dir'], 'met'),
                    'RRAP23_met' : (['rev3_atmos'], '2023_met_heron', ['TEMP','PRESS', 'CSPEED', 'CDIR'], ['m01s16i004','m01s00i408','wind_speed','wind_dir'], 'met'),
                    'EcoRRAP22_davies_combined' : (['rev3_ocean'], '2022_ecorrap_davies', ['LF_TEMP_degrees_Celsius','FS_TEMP_degrees_Celsius','FD_TEMP_degrees_Celsius','FD_PAR_umole_m-2_s-1'], ['temp','PAR'], 'seawater'),
                    'EcoRRAP23_davies_combined' : (['rev3_ocean'], '2023_ecorrap_davies', ['LF_TEMP_degrees_Celsius','FS_TEMP_degrees_Celsius','FD_TEMP_degrees_Celsius','FD_PAR_umole_m-2_s-1'], ['temp','PAR'], 'seawater'),
                    'EcoRRAP22_heron_combined'  : (['rev3_ocean'], '2022_ecorrap_heron', ['LS_TEMP_degrees_Celsius','FS_TEMP_degrees_Celsius','FD_PAR_umole_m-2_s-1'], ['temp','PAR'], 'seawater'),
                    'EcoRRAP23_heron_combined'  : (['rev3_ocean'], '2023_ecorrap_heron', ['FS_TEMP_degrees_Celsius','FD_TEMP_degrees_Celsius','FD_PAR_umole_m-2_s-1'], ['temp','PAR'], 'seawater')}

for obs_name, (model_run, model_name, obs_vars, model_vars, fname_suffix) in comparison_dsets.items():

    obs_tmp = obs_dsets[obs_name]
    out_df = pd.DataFrame({})
    
    for irun in model_run:
        
        if 'atmos' in irun:       
            model_tmp = model_processed[irun][model_name]
            model_aligned, obs_aligned = align_times(model_tmp, obs_tmp) 
            out_df['time'] = model_aligned.time.values
            for var in obs_vars:
                out_df['obs_'+var] = obs_aligned[var].values
            for var in model_vars:
                out_df[irun+'_'+var] = model_aligned[var].values

        elif 'ocean' in irun:
            for i,iloc in enumerate(model_processed[irun][model_name].keys()):
                model_tmp = model_processed[irun][model_name][iloc]  # Davies LF, FS, FD etc
                model_aligned, obs_aligned = align_times(model_tmp, obs_tmp)
                out_df['time'] = model_aligned.time.values
                if i == 0:
                    for var in obs_vars:
                        out_df['obs_'+var] = obs_aligned[var].values
                for var in model_vars:
                    if var in model_aligned.variables:
                        out_df[irun+'_'+iloc+'_'+var] = model_aligned[var].values

    # Write to csv
    out_fname = out_dir+model_name+'_'+fname_suffix+'.csv'
    out_df.to_csv(out_fname, index=False)
    print(f'Saved {out_fname}')
    
    
# ************************************************************************
# Now calculate and format observed and modelled aerosol size distribution
# ************************************************************************

# Save separately as an xr.dataset

# First align timestamps
obs_df = obs_dsets['RRAP23_SD']
model_ds = model_processed['rev3_atmos']['2023_rrap_heron']
model_aligned, obs_aligned = align_times(model_ds, obs_df)

# Format the observed aerosol size distribution
n_obs = len(obs_aligned)
df_headers = obs_aligned.columns.tolist()
idx_diam1 = df_headers.index('Dia1')
idx_diam2 = df_headers.index('Dia150')
idx_conc1 = df_headers.index('Conc1')
idx_conc2 = df_headers.index('Conc150')
obs_diam = np.array(obs_aligned.iloc[0,idx_diam1:idx_diam2]).astype(float) # array of particle sizes (1 x diameter)
obs_conc = np.zeros((len(obs_diam), n_obs))
for t in range(0,n_obs):
    obs_conc[:,t] = obs_aligned.iloc[t,idx_conc1:idx_conc2] # array of particle concentration (diameter x time)
is_missing = np.isnan(obs_diam) # Remove NaNs
obs_diam = obs_diam[~is_missing]
obs_conc = obs_conc[~is_missing, :]

obs_time = obs_aligned.index
SD_out = xr.Dataset({'obs_conc' : (['diameter', 'time'], obs_conc)}, coords = {'diameter' : obs_diam, 'time' : obs_time})

# Now calculate the modelled size distribution
sd_bins = SD_out.diameter.values *1e-9 # size distribution bin diameters in m
for irun in ['control_atmos','rev1_atmos','rev2_atmos','rev3_atmos']:
    model_ds = model_processed[irun]['2023_rrap_heron']
    if irun not in ['control_atmos', 'rev3_atmos']:
        air_temp = model_ds['m01s00i004']
        air_press = model_ds['m01s00i408']
        air_density = calc_air_density(air_press, air_temp)
        model_ds = convert_aer_number_in_mode(model_ds, air_density) # convert aerosol NMR to cm-3 (already done at runtime for control and rev3)
    model_aligned, dummy = align_times(model_ds, obs_aligned)  # obs is already aligned above
    model_SD = construct_size_dist(model_aligned, sd_bins, l_setbins=True)
    SD_out[irun+'_conc'] = xr.DataArray(model_SD['sizedist'], dims=['diameter','time'], coords={'diameter':obs_diam, 'time':obs_time}, name=irun+'_conc')
    
# Write to netcdf
out_fname = out_dir+'2023_rrap_heron_aerosol_SizeDist.nc'
SD_out.to_netcdf(out_fname)
print(f'Saved {out_fname}')

            
print('Done.')
