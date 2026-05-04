'''
Functions used to process and calculate aerosol fields
_______________________________________________________________

calc_air_number_density    : Calculate air density
convert_aerosol_mmr        : Convert aerosol mass from MMR (kg/kg) -> vol concentration (ug/m3).
convert_aer_number_in_mode : Convert aerosol number density from particles/mol -> particles/cm3.
lognormal_culutaive_to_r   : Integrate aerosol number concentration up to a given size.
calc_N_gt_r                : Calculate aerosol number concentration w. dry radius > r.
calc_kohler                : Calculate activation dry diameter at a given supersaturation (from S. Fiddes: https://github.com/sfiddes/ACCESS_aerosol_eval)
calc_ccn                   : Calculate CCN number concentration at a given supersaturation.
construct_size_dist        : Reconstruct aerosol size distribution, using calculate_size_dist & lognormal_dndlogd.
calculate_size_dist        : Evaluate dndlogd for each bin.
lognormal_dndlogd          : Calculate lognormal size distribution at a given diameter.
'''

import xarray as xr
import numpy as np
import scipy


# --------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------
Rd = 287.05 # specific gas constant
cp = 1005.46 # J/kg/K
zboltz = 1.3807e-23 # Boltzmans constant
avo = 6.022e23 # Avogadros number
mm_da = avo * zboltz / Rd # molecular mass of dry air (kg/mol)

# --------------------------------------------------------------------------------------------------
# Calculate air density
# --------------------------------------------------------------------------------------------------
def calc_air_number_density(air_pres, air_temp):
    air_num_dens = air_pres / (air_temp * zboltz * 1.0e6)
    air_num_dens = air_num_dens.assign_attrs({'description':'air number density',
                                              'units':'cm-3'})
    return air_num_dens

# --------------------------------------------------------------------------------------------------
# Convert aerosol MMR (kg/kg air) to mass concentration (ug/m3)
# --------------------------------------------------------------------------------------------------
def convert_aerosol_mmr(aer_ds, air_num_dens, spec):
    if spec == 'so4':
        stash = ['m01s34i102','m01s34i104','m01s34i108','m01s34i114']
    elif spec == 'ss':
        if 'm01s34i127' in list(aer_ds.keys()):
            stash = ['m01s34i111','m01s34i117','m01s34i127'] # include Aitken SS
        else:
            stash = ['m01s34i111','m01s34i117']
    elif spec == 'oc':
        stash = ['m01s34i126','m01s34i106','m01s34i121','m01s34i110','m01s34i116']
    elif spec == 'bc':
        stash = ['m01s34i105','m01s34i109','m01s34i115','m01s34i120']

    #conversion_factor = air_num_dens * mm_da / avo * 1e6        # convert to kg/m3
    #conversion_factor = air_num_dens * mm_da / avo * 1e6 * 1e3  # convert to g/m3
    conversion_factor = air_num_dens * mm_da / avo * 1e6 * 1e9   # convert to ug/m3
    
    for istash in stash:
        meta = aer_ds[istash].attrs
        aer_ds[istash] = aer_ds[istash] * conversion_factor
        aer_ds[istash] = aer_ds[istash].assign_attrs(meta)
        aer_ds[istash] = aer_ds[istash].assign_attrs({'units':'ug m-3'})

    return aer_ds

# --------------------------------------------------------------------------------------------------
# Convert aerosol number densities from particles mol-1 to cm-3
# --------------------------------------------------------------------------------------------------
def convert_aer_number_in_mode(aer_mode_num, air_dens):
    aer_num = ['m01s34i101','m01s34i103','m01s34i107','m01s34i113','m01s34i119'] # soluble nucleation, Aitken, accumulation, coarse, insol Aitken  
    for imode in aer_num:
        aer_mode_num[imode] = aer_mode_num[imode] * air_dens
        aer_mode_num[imode] = aer_mode_num[imode].assign_attrs({'units':'cm-3'})
    return aer_mode_num

# --------------------------------------------------------------------------------------------------
# Calculate aerosol number concentration > r
# --------------------------------------------------------------------------------------------------
def lognormal_cumulative_to_r(N,r,rbar,sigma):
    total_to_r=(N/2.0)*(1.0+scipy.special.erf(np.log(r/rbar)/np.sqrt(2.0)/np.log(sigma)))
    return total_to_r

def calc_N_gt_r(cutoff_r, aer_ds):
    sigma_g = [1.59, 1.59, 1.4, 2.0, 1.59, 1.59, 2.0]
    nd = xr.concat([aer_ds['m01s34i101'], aer_ds['m01s34i103'], aer_ds['m01s34i107'], aer_ds['m01s34i113'], aer_ds['m01s34i119']], "mode")
    rbardry = xr.concat([aer_ds['m01s38i401'], aer_ds['m01s38i402'], aer_ds['m01s38i403'], aer_ds['m01s38i404'], aer_ds['m01s38i405']], "mode")
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
    nd_gt_r = xr.DataArray(nd_gt_r, dims=list(aer_ds['m01s34i101'].dims),
                           attrs={'description':'Calculated aerosol number concentration','units':'cm-3'})
    return nd_gt_r

# --------------------------------------------------------------------------------------------------
# Calculate activation dry diameter at given supersaturation (%)
# --------------------------------------------------------------------------------------------------
# From S. Fiddes: https://github.com/sfiddes/ACCESS_aerosol_eval
def calc_kohler(ss):
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
    
    print('Activation diameter',2*act_r*1e9,'nm')
    return(2*act_r*1e9)

# --------------------------------------------------------------------------------------------------
# Calculate CCN number concentration at a give SS
# --------------------------------------------------------------------------------------------------
def calc_ccn(ds, ss):
    nd = xr.concat([ds['m01s34i101'], ds['m01s34i103'], ds['m01s34i107'], ds['m01s34i113']], "mode")
    rbardry = xr.concat([ds['m01s38i401'], ds['m01s38i402'], ds['m01s38i403'], ds['m01s38i404']], "mode")
    rbardry = rbardry / 2 # convert from diameter -> radius
    
    sigma_g = [1.59, 1.59, 1.4, 2.0, 1.59, 1.59, 2.0]
    nmodes = nd.shape[0]
    cutoff_d = calc_kohler(ss)
    cutoff_r = (cutoff_d/2) *1e-9
    print('Activation diameter', cutoff_d, 'nm')
          
    # loop over number of modes
    for imode in range(nmodes):
        nd_lt_r_this_mode = lognormal_cumulative_to_r(nd[imode], cutoff_r, rbardry[imode], sigma_g[imode])
        nd_gt_r_this_mode = nd[imode] - nd_lt_r_this_mode
        if (imode == 0):
            nd_gt_r = nd_gt_r_this_mode
        else:
            nd_gt_r = nd_gt_r + nd_gt_r_this_mode

    nd_gt_r = xr.DataArray(nd_gt_r, name=f'CCN_{ss}SS)', attrs={'units':'cm-3'})
    
    return nd_gt_r

# --------------------------------------------------------------------------------------------------
# Construct aerosol size distribution
# --------------------------------------------------------------------------------------------------

# Adapted from S. Fiddes: https://github.com/sfiddes/ACCESS_aerosol_eval
# ds  : xr.Dataset w. modelled aerosol number concentration and diameter of each mode
# bins : # of bins to interpolate model output into, or an array of specified bins (set l_setbins to True)
# Optional args
#    l_setbins : True if 'bins' is an array of set bin diameters. False if 'bins' is a number of bins to derive.
#    rmin / rmax : Smallest / largest radius (m) to derive bin diameters if l_setbins = False.

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
    ds_out['sizedist'] = ds_out['sizedist'].assign_attrs({'Units':'dN / dlogD (cm-3)'})
    ds_out['diameter'] = ds_out['diameter'].assign_attrs({'Units':'nm'})
    
    return ds_out

# --------------------------------------------------------------------------------------------------
# Evaluate dndlogd for each size bin
# --------------------------------------------------------------------------------------------------

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
        mode[imode] = np.isfinite(nd[imode,:].any())
    
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


# --------------------------------------------------------------------------------------------------
# Calculate lognormal distribution (dn/dlogd) at diameter d
# --------------------------------------------------------------------------------------------------

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