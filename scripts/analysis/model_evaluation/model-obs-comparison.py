# *********************************************************************
# Compare ACCESS-EMS-GBR model output to observations
# *********************************************************************

# Comparison datasets:

# RRAP 2022 RV Guardian + RV Magnetic ship track: N10, CCN
# RRAP 2023 Heron Island Research Station: N10, CCN, aerosol size distribution, aerosol masses 
# AERONET Lucinda: AOD
# AIMS Weather Stations, Davies Reef and Heron Island: air temp, air pressure, winds
# EcoRRAP seawater loggers, Davies Reef and Heron Island: seawater temperature and PAR


import xarray as xr
from xarray.coding.times import CFTimedeltaCoder
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import cmocean
import cartopy as cp
import cartopy.crs as ccrs
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, RRuleLocator, rrulewrapper
from dateutil.rrule import rrule, DAILY
from tabulate import tabulate
import scipy
from scipy import stats
from sklearn.metrics import r2_score

from functions import *


# *********************************************************************
# Inputs
# *********************************************************************

# Path to preprocessed files
data_dir = '/g/data/p66/rj9627/UM_NS/output/RRAP/obs_comparison/'
files = {'RRAP22_N10'       : '2022_rrap_ship_aerosol_N10.csv',
         'RRAP22_CCN'       : '2022_rrap_ship_aerosol_CCN.csv',
         'RRAP23_N10'       : '2023_rrap_heron_aerosol_N10.csv',
         'RRAP23_CCN'       : '2023_rrap_heron_aerosol_CCN.csv',
         'RRAP23_SizeDist'  : '2023_rrap_heron_aerosol_SizeDist.nc',
         'RRAP23_mass'      : '2023_rrap_heron_aerosol_mass.csv',
         'AERONET22'        : '2022_aeronet_aerosol_AOD.csv',
         'AERONET23'        : '2023_aeronet_aerosol_AOD.csv',
         'AIMS22_Davies'    : '2022_aims_davies_met.csv',
         'AIMS23_Davies'    : '2023_aims_davies_met.csv',
         'AIMS22_Heron'     : '2022_aims_heron_met.csv',
         'RRAP23_met'       : '2023_met_heron_met.csv',
         'EcoRRAP22_Davies' : '2022_ecorrap_davies_seawater.csv',
         'EcoRRAP23_Davies' : '2023_ecorrap_davies_seawater.csv',
         'EcoRRAP22_Heron'  : '2022_ecorrap_heron_seawater.csv',
         'EcoRRAP23_Heron'  : '2023_ecorrap_heron_seawater.csv'}

print('Reading datasets')
dsets = {}
for ds_name,ifile in files.items():
    if ifile.endswith('.csv'):
        dsets[ds_name] = pd.read_csv(data_dir+ifile, parse_dates=['time'])
    elif ifile.endswith('.nc'):
        dsets[ds_name] = xr.open_dataset(data_dir+ifile)
    print(f'    Read {ifile}')
    

# Plot settings
fontsize = 18
plt.rcParams.update({'font.size':fontsize, 'mathtext.default':'regular'})
date_form = DateFormatter("%d-%m-%Y")
line_settings = {'obs'  : {'linestyle':'', 'color':'k', 'marker':'.', 'markersize':10, 'linewidth':1},
                 'ctl'  : {'linestyle':'', 'color':'slategrey', 'marker':'.', 'markersize':10, 'linewidth':1, 'alpha':0.7},   # Control run
                 'rev1' : {'linestyle':'', 'color':'darkorange', 'marker':'.', 'markersize':10, 'linewidth':1, 'alpha':0.7},  # +Ait SS +EDGAR
                 'rev2' : {'linestyle':'', 'color':'forestgreen', 'marker':'.', 'markersize':10, 'linewidth':1, 'alpha':0.7}, # +Ait SS +EDGAR +BLN +monoterp
                 'rev3' : {'linestyle':'', 'color':'C0', 'marker':'.', 'markersize':10, 'linewidth':1, 'alpha':0.7}}          # +Ait SS +EDGAR +BLN +monoterp +SOA scaling +PMOC size change
plt_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
fig_dir = '/home/578/rj9627/python/shared/access-ems-gbr-evaluation/plots/'


# *********************************************************************
# N10 and CCN (Fig. 3)
# *********************************************************************

print('Comparing N10 and CCN')

# N10
N10_22 = dsets['RRAP22_N10']
N10_23 = dsets['RRAP23_N10']
N10_22_obs = N10_22['obs_CN_combined']
N10_23_obs = N10_23['obs_conc']
N10_22_model = N10_22['rev3_atmos_N10']
N10_23_model = N10_23['rev3_atmos_N10']

# CCN
ss = 0.1
CCN_22 = dsets['RRAP22_CCN']
CCN_23 = dsets['RRAP23_CCN']
CCN_22_obs = CCN_22['obs_CCN_0.1']
CCN_23_obs = CCN_23['obs_CCN_0.1']
CCN_22_model = CCN_22['rev3_atmos_CCN_0.1']
CCN_23_model = CCN_23['rev3_atmos_CCN_0.1']

# Plot
nrows = 2
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize=(15*ncols,4*nrows), sharex='col')

axs[0,0].plot(N10_22['time'], N10_22_obs, **line_settings['obs'], label='Observed')
axs[0,0].plot(N10_22['time'], N10_22_model, **line_settings['rev3'], label='Modelled')
axs[0,0].set_ylabel('N10 (cm$^{-3}$)')

axs[0,1].plot(N10_23.time, N10_23_obs, **line_settings['obs'], label='Observed')
axs[0,1].plot(N10_23.time, N10_23_model, **line_settings['rev3'], label='Modelled')
axs[0,1].set_ylabel('N10 (cm$^{-3}$)')

axs[1,0].plot(CCN_22.time, CCN_22_obs, **line_settings['obs'], label='Observed')
axs[1,0].plot(CCN_22.time, CCN_22_model, **line_settings['rev3'], label='Modelled')
axs[1,0].set_ylabel('CCN 0.1% SS (cm$^{-3}$)')

axs[1,1].plot(CCN_23.time, CCN_23_obs, **line_settings['obs'], label='Observed')
axs[1,1].plot(CCN_23.time, CCN_23_model, **line_settings['rev3'], label='Modelled')
axs[1,1].set_ylabel('CCN 0.1% SS (cm$^{-3}$)')

for i in range(nrows*ncols):
    r = i // ncols
    c = i % ncols
    iax = axs[r,c]
    iax.annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.94,0.83), color='black', fontsize=fontsize*1.75)
    plot_start = mdates.num2date(iax.get_xlim()[0]).replace(tzinfo=None)
    start_dt = plot_start.replace(hour=2, minute=0, second=0, microsecond=0)      # label 02:00 UTC (12:00 UTC+10)
    xtick_rule = rrulewrapper(freq=DAILY, interval=2, byhour=2, dtstart=start_dt) # every 2nd day
    xtick_loc = RRuleLocator(xtick_rule)
    iax.xaxis.set_major_locator(xtick_loc)
    iax.xaxis.set_major_formatter(date_form)
    if r == nrows-1:
        iax.tick_params('x', labelrotation=45)
        iax.set_xlabel('Date (noon UTC+10)')

axs[0,1].legend(loc='upper left', framealpha=0.5, markerscale=2)
axs[0,0].set_ylim([0.1, 6000])
axs[0,1].set_ylim([0.1, 4000])
plt.subplots_adjust(wspace=0.2)
fout = fig_dir + 'N10_CCN_tseries.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()

# Summary and bias metrics {variable : (obs, [model_runs]}
comparisons = {'N10 (RRAP 2022)' : (N10_22_obs, [N10_22['control_atmos_N10'], N10_22['rev1_atmos_N10'], N10_22['rev2_atmos_N10'], N10_22['rev3_atmos_N10']]),
               'N10 (RRAP 2023)' : (N10_23_obs, [N10_23['control_atmos_N10'], N10_23['rev1_atmos_N10'], N10_23['rev2_atmos_N10'], N10_23['rev3_atmos_N10']]),
               'CCN (RRAP 2022)' : (CCN_22_obs, [CCN_22['control_atmos_CCN_0.1'], CCN_22['rev1_atmos_CCN_0.1'], CCN_22['rev2_atmos_CCN_0.1'], CCN_22['rev3_atmos_CCN_0.1']]),
               'CCN (RRAP 2023)' : (CCN_23_obs, [CCN_23['control_atmos_CCN_0.1'], CCN_23['rev1_atmos_CCN_0.1'], CCN_23['rev2_atmos_CCN_0.1'], CCN_23['rev3_atmos_CCN_0.1']])}

df = generate_summary_table(comparisons)
fout = fig_dir + 'N10_CCN_stats.csv'
df.to_csv(fout, index=False)


# *********************************************************************
# Aerosol size distribution (Fig. 4)
# *********************************************************************

print('Comparing aerosol size distribution')

SDist = dsets['RRAP23_SizeDist']
bins = SDist['diameter'].values
Nd_obs = SDist['obs_conc'].values
Nd_model = SDist['rev3_atmos_conc'].values

# Plot time-averaged size distribution
Nd_obs_avg = np.nanmean(Nd_obs, axis=1)
Nd_model_avg = np.nanmean(Nd_model, axis=1)
p10_obs = np.nanpercentile(Nd_obs, 10, axis=1)
p90_obs = np.nanpercentile(Nd_obs, 90, axis=1)
fig, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(bins, Nd_obs_avg, **line_settings['obs'], label='Observed')
ax.plot(bins, Nd_model_avg, **line_settings['rev3'], label='Modelled')
ax.fill_between(bins, p10_obs, p90_obs, color='grey', alpha=0.2, label='_nolabel')
ax.set(xlabel='Dp (nm)', ylabel='dN/dlogDp (cm$^{-3}$)')
ax.set_xscale('log')
ax.legend(loc='best', framealpha=0.5, fontsize=12, markerscale=2)
fout = fig_dir + 'Aerosol_SizeDist_avg.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()

# Plot time-series
cmap = cmocean.cm.haline
cNorm = colors.LogNorm(vmin=1, vmax=10000)
nrows = 2
ncols = 1
fig, axs = plt.subplots(nrows, ncols, figsize=(15*ncols,4*nrows),sharex=True)
im_ratio = (4*nrows) / (15*ncols)
obs_plt = axs[0].pcolormesh(SDist.time, bins, Nd_obs, shading='auto', norm=cNorm, cmap=cmap)
model_plt = axs[1].pcolormesh(SDist.time, bins, Nd_model, shading='auto', norm=cNorm, cmap=cmap) 
for i in range(nrows*ncols):
    axs[i].set_yscale('log')
    axs[i].set(ylabel='Dp (nm)')
    axs[i].tick_params(axis='both', which='major')
    
    plot_start = mdates.num2date(iax.get_xlim()[0]).replace(tzinfo=None)
    start_dt = plot_start.replace(hour=2, minute=0, second=0, microsecond=0)
    xtick_rule = rrulewrapper(freq=DAILY, interval=2, byhour=2, dtstart=start_dt)
    xtick_loc = RRuleLocator(xtick_rule)
    iax.xaxis.set_major_locator(xtick_loc)
    iax.xaxis.set_major_formatter(date_form)
    axs[i].tick_params('x', labelrotation=45)
    axs[i].annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.93, 0.82), color='white', fontsize=fontsize*1.75)
axs[-1].set(xlabel='Date (noon UTC+10)')
plt.colorbar(obs_plt, ax=axs[:], fraction=0.046*im_ratio, orientation='vertical').set_label('dN/dlogDp (cm$^{-3}$)')
fout = fig_dir + 'Aerosol_SizeDist_tseries.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()


# *********************************************************************
# Observed v modelled CCN for all measured SS (Fig. 5)
# *********************************************************************

print('Comparing CCN across all measured SS')

CCN_22 = dsets['RRAP22_CCN']
CCN_22_comparisons = {'0.1' : (CCN_22['obs_CCN_0.1'], CCN_22['rev3_atmos_CCN_0.1']),
                      '0.2' : (CCN_22['obs_CCN_0.2'], CCN_22['rev3_atmos_CCN_0.2']),
                      '0.3' : (CCN_22['obs_CCN_0.3'], CCN_22['rev3_atmos_CCN_0.3']),
                      '0.5' : (CCN_22['obs_CCN_0.5'], CCN_22['rev3_atmos_CCN_0.5']),
                      '0.7' : (CCN_22['obs_CCN_0.7'], CCN_22['rev3_atmos_CCN_0.7'])}

CCN_23 = dsets['RRAP23_CCN']
CCN_23_comparisons = {'0.1' : (CCN_23['obs_CCN_0.1'], CCN_23['rev3_atmos_CCN_0.1']),
                      '0.2' : (CCN_23['obs_CCN_0.2'], CCN_23['rev3_atmos_CCN_0.2']),
                      '0.3' : (CCN_23['obs_CCN_0.3'], CCN_23['rev3_atmos_CCN_0.3']),
                      '0.5' : (CCN_23['obs_CCN_0.5'], CCN_23['rev3_atmos_CCN_0.5']),
                      '0.7' : (CCN_23['obs_CCN_0.7'], CCN_23['rev3_atmos_CCN_0.7'])}

# Scatter plot
nrows = 1
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols,7*nrows), layout='constrained', sharey=True)

# 2022
obs_combined = []
model_combined = []
for i_ss,(iobs,imodel) in CCN_22_comparisons.items():
    is_valid = iobs.notna() & imodel.notna()
    iobs = iobs[is_valid]
    imodel = imodel[is_valid]
    axs[0].plot(iobs, imodel, '.', label=f'{i_ss}%')
    obs_combined.extend(iobs)
    model_combined.extend(imodel)
r_combined = np.corrcoef(obs_combined, model_combined)[0,1]
r2_combined = r_combined **2
axs[0].text(0.05, 0.95, f'$R^2$ = {r2_combined:.2f}', transform=axs[0].transAxes, va='top')
axs[0].set_xlim([20,1200])
axs[0].set_ylim([20,1200])
axs[0].plot([20, 1200], [20, 1200], 'k--', lw=1, label='_nolabel')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set(xlabel='Observed (cm$^{-3}$)', ylabel='Modelled (cm$^{-3}$)')
axs[0].set_title('February 2022')
axs[0].legend(loc='lower right', framealpha=0.5, markerscale=4)

# 2023
obs_combined = []
model_combined = []
for i_ss,(iobs,imodel) in CCN_23_comparisons.items():
    is_valid = iobs.notna() & imodel.notna()
    iobs = iobs[is_valid]
    imodel = imodel[is_valid]
    axs[1].plot(iobs, imodel, '.', label=f'{i_ss}%')
    obs_combined.extend(iobs)
    model_combined.extend(imodel)
r_combined = np.corrcoef(obs_combined, model_combined)[0,1]
r2_combined = r_combined **2
axs[1].text(0.05, 0.95, f'$R^2$ = {r2_combined:.2f}', transform=axs[1].transAxes, va='top')
axs[1].set_xlim([20,1200])
axs[1].set_ylim([20,1200])
axs[1].plot([20, 1200], [20, 1200], 'k--', lw=1, label='_nolabel')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set(xlabel='Observed (cm$^{-3}$)')
axs[1].set_title('March 2023')

fout = fig_dir + 'CCN_scatter.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()


# *********************************************************************
# Aerosol sulfate, organics and black carbon masses (Fig. 6)
# *********************************************************************

print('Comparing aerosol masses')

mass_ds = dsets['RRAP23_mass']

sul_obs = mass_ds['obs_HRSO4']
org_obs = mass_ds['obs_HROrg']
bc_obs = mass_ds['obs_BC2;'] *1e-3 # ng/m3 –> ug/m3

sul_model = mass_ds['rev3_atmos_aerosol_so4_mass']
org_model = mass_ds['rev3_atmos_aerosol_org_mass']
bc_model = mass_ds['rev3_atmos_aerosol_bc_mass']

# Filter observed BC >= 50 ng/m3 (flagged as possible ship exhaust)
bc_obs = np.where(bc_obs >= 0.05, np.nan, bc_obs)

# Apply the same filtering to the model for consistency (handle cases where elevated BC is not actually ship exhaust)
bc_model = np.where(bc_model >= 0.05, np.nan, bc_model)

# Plot
nrows = 3
ncols = 1
fig, axs = plt.subplots(nrows, ncols, figsize=(12*ncols,4*nrows), sharex='col')

axs[0].plot(mass_ds['time'], sul_obs, **line_settings['obs'], label='Observed')
axs[0].plot(mass_ds['time'], sul_model, **line_settings['rev3'], label='Modelled')
axs[0].set_ylabel('sulphate (ug m$^{-3}$)')

axs[1].plot(mass_ds['time'], org_obs, **line_settings['obs'], label='Observed')
axs[1].plot(mass_ds['time'], org_model, **line_settings['rev3'], label='Modelled')
axs[1].set_ylabel('organics (ug m$^{-3}$)')

axs[2].plot(mass_ds['time'], bc_obs, **line_settings['obs'], label='Observed')
axs[2].plot(mass_ds['time'], bc_model, **line_settings['rev3'], label='Modelled')
axs[2].set_ylabel('black carbon (ug m$^{-3}$)')

plot_start = mdates.num2date(axs[0].get_xlim()[0]).replace(tzinfo=None)
start_dt = plot_start.replace(hour=2, minute=0, second=0, microsecond=0)
xtick_rule = rrulewrapper(freq=DAILY, interval=2, byhour=2, dtstart=start_dt)
xtick_loc = RRuleLocator(xtick_rule)
for i in range(nrows):
    axs[i].annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.94,0.83), color='black', fontsize=fontsize*1.75)
    axs[i].xaxis.set_major_locator(xtick_loc)
    axs[i].xaxis.set_major_formatter(date_form)
    if i == nrows-1:
        axs[i].tick_params('x', labelrotation=45)
        axs[i].set_xlabel('Date (noon UTC+10)')
    
    axs[0].legend(loc='upper left', framealpha=0.5, markerscale=2)
    plt.subplots_adjust(wspace=0.2)
    fout = fig_dir + 'aerosol_mass_tseries.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()

# Summary and bias metrics {variable : (obs, [model_runs]}
model_bc_vars = ['control_atmos_aerosol_bc_mass', 'rev1_atmos_aerosol_bc_mass', 'rev2_atmos_aerosol_bc_mass', 'rev3_atmos_aerosol_bc_mass'] # Apply BC filtering to all model runs
for i,ivar in enumerate(model_bc_vars):
    mass_ds[ivar] = np.where(mass_ds[ivar] >= 0.05, np.nan, mass_ds[ivar])
    
comparisons = {'Aerosol sulfate mass'      : (sul_obs, [mass_ds['control_atmos_aerosol_so4_mass'], mass_ds['rev1_atmos_aerosol_so4_mass'], mass_ds['rev2_atmos_aerosol_so4_mass'], mass_ds['rev3_atmos_aerosol_so4_mass']]),
               'Aerosol organic mass'      : (org_obs, [mass_ds['control_atmos_aerosol_org_mass'], mass_ds['rev1_atmos_aerosol_org_mass'], mass_ds['rev2_atmos_aerosol_org_mass'], mass_ds['rev3_atmos_aerosol_org_mass']]),
               'Aerosol black carbon mass' : (bc_obs, [mass_ds['control_atmos_aerosol_bc_mass'], mass_ds['rev1_atmos_aerosol_bc_mass'], mass_ds['rev2_atmos_aerosol_bc_mass'], mass_ds['rev3_atmos_aerosol_bc_mass']])}
    
df = generate_summary_table(comparisons)
fout = fig_dir + 'aerosol_mass_stats.csv'
df.to_csv(fout, index=False)


# *********************************************************************
# AERONET AOD (Fig. 7)
# *********************************************************************

print('Comparing AOD')

AOD_22 = dsets['AERONET22']
AOD_23 = dsets['AERONET23']

AOD_22_obs = AOD_22['obs_AOD_551nm']
AOD_23_obs = AOD_23['obs_AOD_532nm']

AOD_22_model = AOD_22['rev3_atmos_AOD']
AOD_23_model = AOD_23['rev3_atmos_AOD']

# Remove invalid obs (-9999)
AOD_22_obs = np.where(AOD_22_obs < 0, np.nan, AOD_22_obs)
AOD_23_obs = np.where(AOD_23_obs < 0, np.nan, AOD_23_obs)

# Plot
nrows = 1
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize=(12*ncols,4*nrows), sharex='col')

axs[0].plot(AOD_22['time'], AOD_22_obs, **line_settings['obs'], label='Observed')
axs[0].plot(AOD_22['time'], AOD_22_model, **line_settings['rev3'], label='Modelled')
axs[1].plot(AOD_23['time'], AOD_23_obs, **line_settings['obs'], label='Observed')
axs[1].plot(AOD_23['time'], AOD_23_model, **line_settings['rev3'], label='Modelled')
axs[0].set_ylabel('AOD')

for i in range(ncols):
    axs[i].annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.02,0.88), color='black', fontsize=fontsize*1.50)
    plot_start = mdates.num2date(axs[i].get_xlim()[0]).replace(tzinfo=None)
    start_dt = plot_start.replace(hour=2, minute=0, second=0, microsecond=0)
    xtick_rule = rrulewrapper(freq=DAILY, interval=3, byhour=2, dtstart=start_dt)
    xtick_loc = RRuleLocator(xtick_rule)
    axs[i].xaxis.set_major_locator(xtick_loc)
    axs[i].xaxis.set_major_formatter(date_form)
    axs[i].tick_params('x', labelrotation=45)
    axs[i].set_xlabel('Date (noon UTC+10)')
    
# Add legend and save
axs[1].legend(loc='best', framealpha=0.5, markerscale=2)
fout = fig_dir + 'AOD_tseries.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()

# Summary and bias metrics {variable : (obs, [model_runs]}
comparisons = {'AOD (2022)' : (AOD_22_obs, [AOD_22['control_atmos_AOD'], AOD_22['rev1_atmos_AOD'], AOD_22['rev2_atmos_AOD'], AOD_22['rev3_atmos_AOD']]),
               'AOD (2023)' : (AOD_23_obs, [AOD_23['control_atmos_AOD'], AOD_23['rev1_atmos_AOD'], AOD_23['rev2_atmos_AOD'], AOD_23['rev3_atmos_AOD']])}

df = generate_summary_table(comparisons)
fout = fig_dir + 'AOD_stats.csv'
df.to_csv(fout, index=False)


# *********************************************************************
# EcoRRAP seawater temperature (Fig. 8)
# *********************************************************************

print('Comparing seawater temperature')

Davies_22 = dsets['EcoRRAP22_Davies']
Davies_23 = dsets['EcoRRAP23_Davies']
Heron_22 = dsets['EcoRRAP22_Heron']
Heron_23 = dsets['EcoRRAP23_Heron']

lines_tmp = {'obs_davies'   : {'linestyle':'', 'color':'k', 'marker':'.', 'markersize':5, 'linewidth':1.5},
             'obs_heron'   : {'linestyle':'--', 'color':'k', 'linewidth':1.5},
             'model_davies' : {'linestyle':'', 'color':'C0', 'marker':'.', 'markersize':5, 'linewidth':1.5},
             'model_heron' : {'linestyle':'--', 'color':'C0', 'linewidth':1.5}}

# Plot
nrows = 3
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize=(12*ncols,4*nrows), sharex='col', sharey='row')

### Lagoon flat/shallow ###
axs[0,0].plot(Davies_22['time'], Davies_22['obs_LF_TEMP_degrees_Celsius'], **lines_tmp['obs_davies'], label='Observed (Davies)')
axs[0,0].plot(Davies_22['time'], Davies_22['rev3_ocean_Davies_LF_temp'], **lines_tmp['model_davies'], label='Modelled (Davies)')
axs[0,0].plot(Heron_22['time'], Heron_22['obs_LS_TEMP_degrees_Celsius'], **lines_tmp['obs_heron'], label='Observed (Heron)')
axs[0,0].plot(Heron_22['time'], Heron_22['rev3_ocean_Heron_LS_temp'], **lines_tmp['model_heron'], label='Modelled (Heron)')
axs[0,0].set_ylabel('Reef lagoon\ntemperature ($^\circ\!$C)')

axs[0,1].plot(Davies_23['time'], Davies_23['obs_LF_TEMP_degrees_Celsius'], **lines_tmp['obs_davies'], label='Observed (Davies)')
axs[0,1].plot(Davies_23['time'], Davies_23['rev3_ocean_Davies_LF_temp'], **lines_tmp['model_davies'], label='Modelled (Davies)')
# No 2023 temperature obs for Heron lagoon shallow site
axs[0,1].set_ylim([25,31])

### Shallow reef front ###
axs[1,0].plot(Davies_22['time'], Davies_22['obs_FS_TEMP_degrees_Celsius'], **lines_tmp['obs_davies'], label='Observed (Davies)')
axs[1,0].plot(Davies_22['time'], Davies_22['rev3_ocean_Davies_FS_temp'], **lines_tmp['model_davies'], label='Modelled (Davies)')
axs[1,0].plot(Heron_22['time'], Heron_22['obs_FS_TEMP_degrees_Celsius'], **lines_tmp['obs_heron'], label='Observed (Heron)')
axs[1,0].plot(Heron_22['time'], Heron_22['rev3_ocean_Heron_FS_temp'], **lines_tmp['model_heron'], label='Modelled (Heron)')
axs[1,0].set_ylabel('Shallow reef front\ntemperature ($^\circ\!$C)')

axs[1,1].plot(Davies_23['time'], Davies_23['obs_FS_TEMP_degrees_Celsius'], **lines_tmp['obs_davies'], label='Observed (Davies)')
axs[1,1].plot(Davies_23['time'], Davies_23['rev3_ocean_Davies_FS_temp'], **lines_tmp['model_davies'], label='Modelled (Davies)')
axs[1,1].plot(Heron_23['time'], Heron_23['obs_FS_TEMP_degrees_Celsius'], **lines_tmp['obs_heron'], label='Observed (Heron)')
axs[1,1].plot(Heron_23['time'], Heron_23['rev3_ocean_Heron_FS_temp'], **lines_tmp['model_heron'], label='Modelled (Heron)')
axs[1,1].set_ylim([25.5,30.5])

### Deep reef front ###
axs[2,0].plot(Davies_22['time'], Davies_22['obs_FD_TEMP_degrees_Celsius'], **lines_tmp['obs_davies'], label='Observed (Davies)')
axs[2,0].plot(Davies_22['time'], Davies_22['rev3_ocean_Davies_FD_temp'], **lines_tmp['model_davies'], label='Modelled (Davies)')
# No 2022 temperature obs for Heron deep front site
axs[2,0].set_ylabel('Deep reef front\ntemperature ($^\circ\!$C)')

axs[2,1].plot(Davies_23['time'], Davies_23['obs_FD_TEMP_degrees_Celsius'], **lines_tmp['obs_davies'], label='Observed (Davies)')
axs[2,1].plot(Davies_23['time'], Davies_23['rev3_ocean_Davies_FD_temp'], **lines_tmp['model_davies'], label='Modelled (Davies)')
axs[2,1].plot(Heron_23['time'], Heron_23['obs_FD_TEMP_degrees_Celsius'], **lines_tmp['obs_heron'], label='Observed (Heron)')
axs[2,1].plot(Heron_23['time'], Heron_23['rev3_ocean_Heron_FD_temp'], **lines_tmp['model_heron'], label='Modelled (Heron)')
axs[2,1].set_ylim([26.5,30])

# Format axes
plot_start_22 = mdates.num2date(axs[0,0].get_xlim()[0]).replace(tzinfo=None)
start_dt_22 = plot_start_22.replace(hour=2, minute=0, second=0, microsecond=0)
xtick_rule_22 = rrulewrapper(freq=DAILY, interval=3, byhour=2, dtstart=start_dt_22)
xtick_loc_22 = RRuleLocator(xtick_rule_22)

plot_start_23 = mdates.num2date(axs[0,1].get_xlim()[0]).replace(tzinfo=None)
start_dt_23 = plot_start_23.replace(hour=2, minute=0, second=0, microsecond=0)
xtick_rule_23 = rrulewrapper(freq=DAILY, interval=3, byhour=2, dtstart=start_dt_23)
xtick_loc_23 = RRuleLocator(xtick_rule_23)

for i in range(nrows*ncols):
    r = i // ncols
    c = i % ncols 
    iax = axs[r,c]
    iax.annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.02,0.83), color='black', fontsize=fontsize*1.75)
    xtick_loc = xtick_loc_22 if c == 0 else xtick_loc_23
    iax.xaxis.set_major_locator(xtick_loc)
    iax.xaxis.set_major_formatter(date_form)
    if r == nrows-1:
        iax.tick_params('x', labelrotation=45)
        iax.set_xlabel('Date (noon UTC+10)')

# Add legend
handles = []
labels = []
for iax in axs.flat:
    h, l = iax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
by_label = dict(zip(labels, handles))
axs[2,0].legend(by_label.values(), by_label.keys(), loc='best', framealpha=0.5, ncols=2, markerscale=2)
plt.subplots_adjust(wspace=0.05, hspace=0.2)

# Save
fout = fig_dir + 'seawater_temperature_tseries.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()

# Summary and bias metrics {variable : (obs, [model_runs]}             
comparisons = {'Seawater temp: 2022 Davies LF' : (Davies_22['obs_LF_TEMP_degrees_Celsius'], [Davies_22['rev3_ocean_Davies_LF_temp']]),
               'Seawater temp: 2022 Davies FS' : (Davies_22['obs_FS_TEMP_degrees_Celsius'], [Davies_22['rev3_ocean_Davies_FS_temp']]),
               'Seawater temp: 2022 Davies FD' : (Davies_22['obs_FD_TEMP_degrees_Celsius'], [Davies_22['rev3_ocean_Davies_FD_temp']]),
               'Seawater temp: 2023 Davies LF' : (Davies_23['obs_LF_TEMP_degrees_Celsius'], [Davies_23['rev3_ocean_Davies_LF_temp']]),
               'Seawater temp: 2023 Davies FS' : (Davies_23['obs_FS_TEMP_degrees_Celsius'], [Davies_23['rev3_ocean_Davies_FS_temp']]),
               'Seawater temp: 2023 Davies FD' : (Davies_23['obs_FD_TEMP_degrees_Celsius'], [Davies_23['rev3_ocean_Davies_FD_temp']]),
               'Seawater temp: 2022 Heron LS' : (Heron_22['obs_LS_TEMP_degrees_Celsius'], [Heron_22['rev3_ocean_Heron_LS_temp']]),
               'Seawater temp: 2022 Heron FS' : (Heron_22['obs_FS_TEMP_degrees_Celsius'], [Heron_22['rev3_ocean_Heron_FS_temp']]),
               'Seawater temp: 2023 Heron FS' : (Heron_23['obs_FS_TEMP_degrees_Celsius'], [Heron_23['rev3_ocean_Heron_FS_temp']]),
               'Seawater temp: 2023 Heron FD' : (Heron_23['obs_FD_TEMP_degrees_Celsius'], [Heron_23['rev3_ocean_Heron_FD_temp']])}

df = generate_summary_table(comparisons)
fout = fig_dir + 'seawater_temp_stats.csv'
df.to_csv(fout, index=False)


# *********************************************************************
# EcoRRAP seawater PAR (Fig. 9)
# *********************************************************************

print('Comparing seawater PAR')

Davies_22 = dsets['EcoRRAP22_Davies']
Davies_23 = dsets['EcoRRAP23_Davies']
Heron_22 = dsets['EcoRRAP22_Heron']
Heron_23 = dsets['EcoRRAP23_Heron']

lines_tmp = {'obs_par' : {'linestyle':'-', 'color':'k', 'linewidth':2},
             'model_par' : {'linestyle':':', 'color':'C0', 'linewidth':2.5}}

# Plot
nrows = 2
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols,3*nrows), sharex='col', sharey='row')

axs[0,0].plot(Davies_22['time'], Davies_22['obs_FD_PAR_umole_m-2_s-1'], **lines_tmp['obs_par'], label='Observed')
axs[0,0].plot(Davies_22['time'], Davies_22['rev3_ocean_Davies_FD_PAR'], **lines_tmp['model_par'], label='Modelled')
axs[0,1].plot(Davies_23['time'], Davies_23['obs_FD_PAR_umole_m-2_s-1'], **lines_tmp['obs_par'], label='Observed')
axs[0,1].plot(Davies_23['time'], Davies_23['rev3_ocean_Davies_FD_PAR'], **lines_tmp['model_par'], label='Modelled')

axs[1,0].plot(Heron_22['time'], Heron_22['obs_FD_PAR_umole_m-2_s-1'], **lines_tmp['obs_par'], label='Observed')
axs[1,0].plot(Heron_22['time'], Heron_22['rev3_ocean_Heron_FD_PAR'], **lines_tmp['model_par'], label='Modelled')
axs[1,1].plot(Heron_23['time'], Heron_23['obs_FD_PAR_umole_m-2_s-1'], **lines_tmp['obs_par'], label='Observed')
axs[1,1].plot(Heron_23['time'], Heron_23['rev3_ocean_Heron_FD_PAR'], **lines_tmp['model_par'], label='Modelled')

# Format axes
plot_start_22 = mdates.num2date(axs[0,0].get_xlim()[0]).replace(tzinfo=None)
start_dt_22 = plot_start_22.replace(hour=2, minute=0, second=0, microsecond=0)
xtick_rule_22 = rrulewrapper(freq=DAILY, interval=3, byhour=2, dtstart=start_dt_22)
xtick_loc_22 = RRuleLocator(xtick_rule_22)

plot_start_23 = mdates.num2date(axs[0,1].get_xlim()[0]).replace(tzinfo=None)
start_dt_23 = plot_start_23.replace(hour=2, minute=0, second=0, microsecond=0)
xtick_rule_23 = rrulewrapper(freq=DAILY, interval=3, byhour=2, dtstart=start_dt_23)
xtick_loc_23 = RRuleLocator(xtick_rule_23)

for i in range(nrows*ncols):
    r = i // ncols
    c = i % ncols 
    iax = axs[r,c]
    iax.annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.02,0.83), color='black', fontsize=fontsize*1.5)
    xtick_loc = xtick_loc_22 if c == 0 else xtick_loc_23
    iax.xaxis.set_major_locator(xtick_loc)
    iax.xaxis.set_major_formatter(date_form)
    if r == nrows-1:
        iax.tick_params('x', labelrotation=45)
        iax.set_xlabel('Date (noon UTC+10)')
    if c == 0:
        iax.set_ylabel('PAR\n(mol m$^{-2}$ hr$^{-1}$)')

axs[-1,-1].legend(loc='best', framealpha=0.5, ncols=2, markerscale=2)
plt.subplots_adjust(wspace=0.05, hspace=0.2)
fout = fig_dir + 'seawater_PAR_tseries.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()

# Summary and bias metrics {variable : (obs, [model_runs]}             
comparisons = {'Seawater PAR: 2022 Davies FD' : (Davies_22['obs_FD_PAR_umole_m-2_s-1'], [Davies_22['rev3_ocean_Davies_FD_PAR']]),
               'Seawater PAR: 2023 Davies FD' : (Davies_23['obs_FD_PAR_umole_m-2_s-1'], [Davies_23['rev3_ocean_Davies_FD_PAR']]),
               'Seawater PAR: 2022 Heron FD' : (Heron_22['obs_FD_PAR_umole_m-2_s-1'], [Heron_22['rev3_ocean_Heron_FD_PAR']]),
               'Seawater PAR: 2023 Heron FD' : (Heron_23['obs_FD_PAR_umole_m-2_s-1'], [Heron_23['rev3_ocean_Heron_FD_PAR']])}

df = generate_summary_table(comparisons)
fout = fig_dir + 'seawater_PAR_stats.csv'
df.to_csv(fout, index=False)


# *********************************************************************************
# Appendix B: Effect of revisions to the aerosol scheme on modelled aerosol biases
# *********************************************************************************

# Create plots for Appendix B, comparing observed aerosol variables to model output
# for the control, rev1, rev2 and rev3 runs. Summary stats and bias metrics for all
# model runs have already been created and saved above.

print('Generating plots for Appendix B')

run_id = ['ctl','rev1','rev2','rev3']
run_labels = ['Control','Ait SS + EDGAR','Ait SS + EDGAR + BLN','Revised']

#### N10 and CCN (Fig. B1) ####

N10_22 = dsets['RRAP22_N10']
N10_23 = dsets['RRAP23_N10']
N10_22_obs = N10_22['obs_CN_combined']
N10_23_obs = N10_23['obs_conc']

CCN_22 = dsets['RRAP22_CCN']
CCN_23 = dsets['RRAP23_CCN']
CCN_22_obs = CCN_22['obs_CCN_0.1']
CCN_23_obs = CCN_23['obs_CCN_0.1']

model_N10_vars = ['control_atmos_N10', 'rev1_atmos_N10', 'rev2_atmos_N10', 'rev3_atmos_N10']
model_CCN_vars = ['control_atmos_CCN_0.1', 'rev1_atmos_CCN_0.1', 'rev2_atmos_CCN_0.1', 'rev3_atmos_CCN_0.1']

# Plot
nrows = 2
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize=(15*ncols,4*nrows), sharex='col')

axs[0,0].plot(N10_22['time'], N10_22_obs, **line_settings['obs'], label='Observed')
axs[0,0].set_ylabel('N10 (cm$^{-3}$)')
for i,ivar in enumerate(model_N10_vars):
    axs[0,0].plot(N10_22['time'], N10_22[ivar], **line_settings[run_id[i]], label=run_labels[i])

axs[0,1].plot(N10_23.time, N10_23_obs, **line_settings['obs'], label='Observed')
axs[0,1].set_ylabel('N10 (cm$^{-3}$)')
for i,ivar in enumerate(model_N10_vars):
    axs[0,1].plot(N10_23['time'], N10_23[ivar], **line_settings[run_id[i]], label=run_labels[i])

axs[1,0].plot(CCN_22.time, CCN_22_obs, **line_settings['obs'], label='Observed')
axs[1,0].set_ylabel('CCN 0.1% SS (cm$^{-3}$)')
for i,ivar in enumerate(model_CCN_vars):
    axs[1,0].plot(CCN_22['time'], CCN_22[ivar], **line_settings[run_id[i]], label=run_labels[i])

axs[1,1].plot(CCN_23.time, CCN_23_obs, **line_settings['obs'], label='Observed')
axs[1,1].set_ylabel('CCN 0.1% SS (cm$^{-3}$)')
for i,ivar in enumerate(model_CCN_vars):
    axs[1,1].plot(CCN_23['time'], CCN_23[ivar], **line_settings[run_id[i]], label=run_labels[i])

for i in range(nrows*ncols):
    r = i // ncols
    c = i % ncols
    iax = axs[r,c]
    iax.annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.94,0.83), color='black', fontsize=fontsize*1.75)
    plot_start = mdates.num2date(iax.get_xlim()[0]).replace(tzinfo=None)
    start_dt = plot_start.replace(hour=2, minute=0, second=0, microsecond=0)      # label 02:00 UTC (12:00 UTC+10)
    xtick_rule = rrulewrapper(freq=DAILY, interval=2, byhour=2, dtstart=start_dt) # every 2nd day
    xtick_loc = RRuleLocator(xtick_rule)
    iax.xaxis.set_major_locator(xtick_loc)
    iax.xaxis.set_major_formatter(date_form)
    if r == nrows-1:
        iax.tick_params('x', labelrotation=45)
        iax.set_xlabel('Date (noon UTC+10)')

axs[0,1].legend(loc='best', framealpha=0.5, markerscale=2, ncols=2)
axs[0,0].set_ylim([0.1, 6000])
axs[0,1].set_ylim([0.1, 4000])
plt.subplots_adjust(wspace=0.2)
fout = fig_dir + 'N10_CCN_tseries_AppendixB.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()


#### Aerosol size distribution (Fig. B2) ####

SDist = dsets['RRAP23_SizeDist']
bins = SDist['diameter'].values
Nd_obs = SDist['obs_conc'].values
model_sd_vars = ['control_atmos_conc', 'rev1_atmos_conc', 'rev2_atmos_conc', 'rev3_atmos_conc']

# Plot time-averaged size distribution
Nd_obs_avg = np.nanmean(Nd_obs, axis=1)
p10_obs = np.nanpercentile(Nd_obs, 10, axis=1)
p90_obs = np.nanpercentile(Nd_obs, 90, axis=1)
fig, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(bins, Nd_obs_avg, **line_settings['obs'], label='Observed')
for i,ivar in enumerate(model_sd_vars):
    Nd_model_avg = np.nanmean(SDist[ivar].values, axis=1)
    ax.plot(bins, Nd_model_avg, **line_settings[run_id[i]], label=run_labels[i])
ax.fill_between(bins, p10_obs, p90_obs, color='grey', alpha=0.2, label='_nolabel')
ax.set(xlabel='Dp (nm)', ylabel='dN/dlogDp (cm$^{-3}$)')
ax.set_xscale('log')
ax.legend(loc='best', framealpha=0.5, fontsize=12, markerscale=2, handletextpad=0.3)
fout = fig_dir + 'Aerosol_SizeDist_avg_AppendixB.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()


#### Aerosol masses (Fig. B3) ####

mass_ds = dsets['RRAP23_mass']

sul_obs = mass_ds['obs_HRSO4']
org_obs = mass_ds['obs_HROrg']
bc_obs = mass_ds['obs_BC2;'] *1e-3 # ng/m3 –> ug/m3

model_sul_vars = ['control_atmos_aerosol_so4_mass', 'rev1_atmos_aerosol_so4_mass', 'rev2_atmos_aerosol_so4_mass', 'rev3_atmos_aerosol_so4_mass']
model_org_vars = ['control_atmos_aerosol_org_mass', 'rev1_atmos_aerosol_org_mass', 'rev2_atmos_aerosol_org_mass', 'rev3_atmos_aerosol_org_mass']
model_bc_vars = ['control_atmos_aerosol_bc_mass', 'rev1_atmos_aerosol_bc_mass', 'rev2_atmos_aerosol_bc_mass', 'rev3_atmos_aerosol_bc_mass']

# Filter observed BC >= 50 ng/m3 (flagged as possible ship exhaust)
bc_obs = np.where(bc_obs >= 0.05, np.nan, bc_obs)

# Apply the same filtering to the model for consistency (handle cases where elevated BC is not actually ship exhaust)
for i,ivar in enumerate(model_bc_vars):
    mass_ds[ivar] = np.where(mass_ds[ivar] >= 0.05, np.nan, mass_ds[ivar])

# Plot
nrows = 3
ncols = 1
fig, axs = plt.subplots(nrows, ncols, figsize=(12*ncols,4*nrows), sharex='col')

axs[0].plot(mass_ds['time'], sul_obs, **line_settings['obs'], label='Observed')
for i,ivar in enumerate(model_sul_vars):
    axs[0].plot(mass_ds['time'], mass_ds[ivar], **line_settings[run_id[i]], label=run_labels[i])
axs[0].set_ylabel('sulphate (ug m$^{-3}$)')

axs[1].plot(mass_ds['time'], org_obs, **line_settings['obs'], label='Observed')
for i,ivar in enumerate(model_org_vars):
    axs[1].plot(mass_ds['time'], mass_ds[ivar], **line_settings[run_id[i]], label=run_labels[i])
axs[1].set_ylabel('organics (ug m$^{-3}$)')

axs[2].plot(mass_ds['time'], bc_obs, **line_settings['obs'], label='Observed')
for i,ivar in enumerate(model_bc_vars):
    axs[2].plot(mass_ds['time'], mass_ds[ivar], **line_settings[run_id[i]], label=run_labels[i])
axs[2].set_ylabel('black carbon (ug m$^{-3}$)')

plot_start = mdates.num2date(axs[0].get_xlim()[0]).replace(tzinfo=None)
start_dt = plot_start.replace(hour=2, minute=0, second=0, microsecond=0)
xtick_rule = rrulewrapper(freq=DAILY, interval=2, byhour=2, dtstart=start_dt)
xtick_loc = RRuleLocator(xtick_rule)
for i in range(nrows):
    axs[i].annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.94,0.83), color='black', fontsize=fontsize*1.75)
    axs[i].xaxis.set_major_locator(xtick_loc)
    axs[i].xaxis.set_major_formatter(date_form)
    if i == nrows-1:
        axs[i].tick_params('x', labelrotation=45)
        axs[i].set_xlabel('Date (noon UTC+10)')
    
axs[1].legend(loc='upper left', framealpha=0.5, markerscale=2, ncol=2)
plt.subplots_adjust(wspace=0.2)
fout = fig_dir + 'aerosol_mass_tseries_AppendixB.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()
    

#### AOD (Fig. B4) ####

AOD_22 = dsets['AERONET22']
AOD_23 = dsets['AERONET23']

AOD_22_obs = AOD_22['obs_AOD_551nm']
AOD_23_obs = AOD_23['obs_AOD_532nm']

# Remove invalid obs (-9999)
AOD_22_obs = np.where(AOD_22_obs < 0, np.nan, AOD_22_obs)
AOD_23_obs = np.where(AOD_23_obs < 0, np.nan, AOD_23_obs)

model_aod_vars = ['control_atmos_AOD', 'rev1_atmos_AOD', 'rev2_atmos_AOD', 'rev3_atmos_AOD']

# Plot
nrows = 1
ncols = 2
fig, axs = plt.subplots(nrows, ncols, figsize=(12*ncols,4*nrows), sharex='col')

axs[0].plot(AOD_22['time'], AOD_22_obs, **line_settings['obs'], label='Observed')
for i,ivar in enumerate(model_aod_vars):
    axs[0].plot(AOD_22['time'], AOD_22[ivar], **line_settings[run_id[i]], label=run_labels[i])

axs[1].plot(AOD_23['time'], AOD_23_obs, **line_settings['obs'], label='Observed')
for i,ivar in enumerate(model_aod_vars):
    axs[1].plot(AOD_23['time'], AOD_23[ivar], **line_settings[run_id[i]], label=run_labels[i])

axs[0].set_ylabel('AOD')
for i in range(ncols):
    axs[i].annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.02,0.88), color='black', fontsize=fontsize*1.75)
    plot_start = mdates.num2date(axs[i].get_xlim()[0]).replace(tzinfo=None)
    start_dt = plot_start.replace(hour=2, minute=0, second=0, microsecond=0)
    xtick_rule = rrulewrapper(freq=DAILY, interval=3, byhour=2, dtstart=start_dt)
    xtick_loc = RRuleLocator(xtick_rule)
    axs[i].xaxis.set_major_locator(xtick_loc)
    axs[i].xaxis.set_major_formatter(date_form)
    axs[i].tick_params('x', labelrotation=45)
    axs[i].set_xlabel('Date (noon UTC+10)')
    
# Add legend and save
axs[1].legend(loc='best', framealpha=0.5, markerscale=2, ncol=2)
plt.subplots_adjust(wspace=0.2)
fout = fig_dir + 'AOD_tseries_AppendixB.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()


# *********************************************************************
# Appendix c: Validation of simulated meteorology (Fig. C1)
# *********************************************************************

print('Generating plots for Appendix C')

met_davies_22 = dsets['AIMS22_Davies']
met_davies_23 = dsets['AIMS23_Davies']
met_heron_22 = dsets['AIMS22_Heron']
met_heron_23 = dsets['RRAP23_met']

# Observations
temp_davies_22_obs = met_davies_22['obs_Air_Temperature']
temp_davies_23_obs = met_davies_23['obs_Air_Temperature']
pres_davies_22_obs = met_davies_22['obs_Air_Pressure']
pres_davies_23_obs = met_davies_23['obs_Air_Pressure']
ws_davies_22_obs = met_davies_22['obs_Wind_Speed_(Scalar_avg_10_min)']
ws_davies_23_obs = met_davies_23['obs_Wind_Speed_(Scalar_avg_10_min)']
wd_davies_22_obs = met_davies_22['obs_Wind_Direction_(Scalar_Average_10_Minutes)']
wd_davies_23_obs = met_davies_23['obs_Wind_Direction_(Scalar_Average_10_Minutes)']

temp_heron_22_obs = met_heron_22['obs_Air_Temperature']
temp_heron_23_obs = met_heron_23['obs_TEMP']
pres_heron_22_obs = met_heron_22['obs_Air_Pressure']
pres_heron_23_obs = met_heron_23['obs_PRESS']
ws_heron_22_obs = met_heron_22['obs_Wind_Speed_(Scalar_avg_30_min)']
ws_heron_23_obs = met_heron_23['obs_CSPEED']
wd_heron_22_obs = met_heron_22['obs_Wind_Direction_(Scalar_Average_30_Minutes)']
wd_heron_23_obs = met_heron_23['obs_CDIR']

# Model
temp_davies_22_model = met_davies_22['rev3_atmos_m01s16i004']
temp_davies_23_model = met_davies_23['rev3_atmos_m01s16i004']
pres_davies_22_model = met_davies_22['rev3_atmos_m01s00i408']
pres_davies_23_model = met_davies_23['rev3_atmos_m01s00i408']
ws_davies_22_model = met_davies_22['rev3_atmos_wind_speed']
ws_davies_23_model = met_davies_23['rev3_atmos_wind_speed']
wd_davies_22_model = met_davies_22['rev3_atmos_wind_dir']
wd_davies_23_model = met_davies_23['rev3_atmos_wind_dir']

temp_heron_22_model = met_heron_22['rev3_atmos_m01s16i004']
temp_heron_23_model = met_heron_23['rev3_atmos_m01s16i004']
pres_heron_22_model = met_heron_22['rev3_atmos_m01s00i408']
pres_heron_23_model = met_heron_23['rev3_atmos_m01s00i408']
ws_heron_22_model = met_heron_22['rev3_atmos_wind_speed']
ws_heron_23_model = met_heron_23['rev3_atmos_wind_speed']
wd_heron_22_model = met_heron_22['rev3_atmos_wind_dir']
wd_heron_23_model = met_heron_23['rev3_atmos_wind_dir']

# Plot
nrows = 4
ncols = 4
fig, axs = plt.subplots(nrows, ncols, figsize=(10*ncols,5*nrows), sharex='col', sharey='row')

# air temperature
axs[0,0].plot(met_heron_22['time'], temp_heron_22_obs, **line_settings['obs'], label='Observed')
axs[0,0].plot(met_heron_22['time'], temp_heron_22_model, **line_settings['rev3'], label='Modelled')
axs[0,1].plot(met_heron_23['time'], temp_heron_23_obs, **line_settings['obs'], label='Observed')
axs[0,1].plot(met_heron_23['time'], temp_heron_23_model, **line_settings['rev3'], label='Modelled')
axs[0,2].plot(met_davies_22['time'], temp_davies_22_obs, **line_settings['obs'], label='Observed')
axs[0,2].plot(met_davies_22['time'], temp_davies_22_model, **line_settings['rev3'], label='Modelled')
axs[0,3].plot(met_davies_23['time'], temp_davies_23_obs, **line_settings['obs'], label='Observed')
axs[0,3].plot(met_davies_23['time'], temp_davies_23_model, **line_settings['rev3'], label='Modelled')
axs[0,0].set_ylabel('Air temperature ($^\circ\!$C)')

# air pressure
axs[1,0].plot(met_heron_22['time'], pres_heron_22_obs, **line_settings['obs'], label='Observed')
axs[1,0].plot(met_heron_22['time'], pres_heron_22_model, **line_settings['rev3'], label='Modelled')
axs[1,1].plot(met_heron_23['time'], pres_heron_23_obs, **line_settings['obs'], label='Observed')
axs[1,1].plot(met_heron_23['time'], pres_heron_23_model, **line_settings['rev3'], label='Modelled')
axs[1,2].plot(met_davies_22['time'], pres_davies_22_obs, **line_settings['obs'], label='Observed')
axs[1,2].plot(met_davies_22['time'], pres_davies_22_model, **line_settings['rev3'], label='Modelled')
axs[1,3].plot(met_davies_23['time'], pres_davies_23_obs, **line_settings['obs'], label='Observed')
axs[1,3].plot(met_davies_23['time'], pres_davies_23_model, **line_settings['rev3'], label='Modelled')
axs[1,0].set_ylabel('Air pressure (hPa)')

# wind speed
axs[2,0].plot(met_heron_22['time'], ws_heron_22_obs, **line_settings['obs'], label='Observed')
axs[2,0].plot(met_heron_22['time'], ws_heron_22_model, **line_settings['rev3'], label='Modelled')
axs[2,1].plot(met_heron_23['time'], ws_heron_23_obs, **line_settings['obs'], label='Observed')
axs[2,1].plot(met_heron_23['time'], ws_heron_23_model, **line_settings['rev3'], label='Modelled')
axs[2,2].plot(met_davies_22['time'], ws_davies_22_obs, **line_settings['obs'], label='Observed')
axs[2,2].plot(met_davies_22['time'], ws_davies_22_model, **line_settings['rev3'], label='Modelled')
axs[2,3].plot(met_davies_23['time'], ws_davies_23_obs, **line_settings['obs'], label='Observed')
axs[2,3].plot(met_davies_23['time'], ws_davies_23_model, **line_settings['rev3'], label='Modelled')
axs[2,0].set_ylabel('Wind speed (m s$^{-1}$)')

# wind direction
axs[3,0].plot(met_heron_22['time'], wd_heron_22_obs, **line_settings['obs'], label='Observed')
axs[3,0].plot(met_heron_22['time'], wd_heron_22_model, **line_settings['rev3'], label='Modelled')
axs[3,1].plot(met_heron_23['time'], wd_heron_23_obs, **line_settings['obs'], label='Observed')
axs[3,1].plot(met_heron_23['time'], wd_heron_23_model, **line_settings['rev3'], label='Modelled')
axs[3,2].plot(met_davies_22['time'], wd_davies_22_obs, **line_settings['obs'], label='Observed')
axs[3,2].plot(met_davies_22['time'], wd_davies_22_model, **line_settings['rev3'], label='Modelled')
axs[3,3].plot(met_davies_23['time'], wd_davies_23_obs, **line_settings['obs'], label='Observed')
axs[3,3].plot(met_davies_23['time'], wd_davies_23_model, **line_settings['rev3'], label='Modelled')
axs[3,0].set_ylabel('Wind direction (degrees)')

# Format axes
for i in range(nrows*ncols):
    r = i // ncols
    c = i % ncols
    iax = axs[r,c]
    iax.annotate(plt_labels[i], xy=(0.1,0.1), xycoords='axes fraction', xytext=(0.94,0.83), color='black', fontsize=fontsize*1.75)
    plot_start = mdates.num2date(iax.get_xlim()[0]).replace(tzinfo=None)
    start_dt = plot_start.replace(hour=2, minute=0, second=0, microsecond=0)      # label 02:00 UTC (12:00 UTC+10)
    xtick_rule = rrulewrapper(freq=DAILY, interval=3, byhour=2, dtstart=start_dt) # every 2nd day
    xtick_loc = RRuleLocator(xtick_rule)
    iax.xaxis.set_major_locator(xtick_loc)
    if r == nrows-1:
        iax.xaxis.set_major_formatter(date_form)
        iax.tick_params('x', labelrotation=45)
        iax.set_xlabel('Date (noon UTC+10)')

# Add legend and save
axs[0,0].legend(loc='best', framealpha=0.5, markerscale=2)
plt.subplots_adjust(wspace=0.05, hspace=0.2)
fout = fig_dir + 'meteorology_tseries_AppendixC.png'
plt.savefig(fout, bbox_inches='tight', dpi=300)
plt.close()

# Summary and bias metrics {variable : (obs, [model_runs]}   
comparisons = {'Heron Island: Air temp 2022'       : (temp_heron_22_obs, [temp_heron_22_model]), 
               'Heron Island: Air temp 2023'       : (temp_heron_23_obs, [temp_heron_23_model]), 
               'Heron Island: Air pressure 2022'   : (pres_heron_22_obs, [pres_heron_22_model]), 
               'Heron Island: Air pressure 2023'   : (pres_heron_23_obs, [pres_heron_23_model]), 
               'Heron Island: Wind speed 2022'     : (ws_heron_22_obs, [ws_heron_22_model]), 
               'Heron Island: Wind speed 2023'     : (ws_heron_23_obs, [ws_heron_23_model]), 
               'Heron Island: Wind direction 2022' : (wd_heron_22_obs, [wd_heron_22_model]), 
               'Heron Island: Wind direction 2023' : (wd_heron_23_obs, [wd_heron_23_model]), 
               
               'Davies Reef: Air temp 2022'       : (temp_davies_22_obs, [temp_davies_22_model]), 
               'Davies Reef: Air temp 2023'       : (temp_davies_23_obs, [temp_davies_23_model]), 
               'Davies Reef: Air pressure 2022'   : (pres_davies_22_obs, [pres_davies_22_model]), 
               'Davies Reef: Air pressure 2023'   : (pres_davies_23_obs, [pres_davies_23_model]), 
               'Davies Reef: Wind speed 2022'     : (ws_davies_22_obs, [ws_davies_22_model]), 
               'Davies Reef: Wind speed 2023'     : (ws_davies_23_obs, [ws_davies_23_model]), 
               'Davies Reef: Wind direction 2022' : (wd_davies_22_obs, [wd_davies_22_model]), 
               'Davies Reef: Wind direction 2023' : (wd_davies_23_obs, [wd_davies_23_model])}

df = generate_summary_table(comparisons)
fout = fig_dir + 'meteorology_stats_AppendixC.csv'
df.to_csv(fout, index=False)

print('Done.')