#!/usr/bin/env python

# Download ERA5 from: https://cds-beta.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=d_download
# Documentation: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

# Download as a single grib file for each month:
# Temperature (t) - units: K, parameter id: 130
# Horizontal U wind (u) - units: m/s, parameter id: 131
# Horizontal V wind (v) - units: m/s, parameter id: 132
# Logarithm of surf pressure (lnsp) - parameter id: 152

# The UM nudging code regrids ERA5 data onto the model grid (bilinear interpolation),
# using LNSP for vertical interpolation of data on hybrid levels.

import cdsapi
import numpy as np
import calendar

start_yr = 2015
start_m = 1
end_yr = 2015
end_m = 1

yr_list = np.arange(start_yr, end_yr+1)
nyrs = len(yr_list)

out_path = '/g/data/p66/rj9627/ERA5/downloads/'

for iyr in range(nyrs):

    if nyrs > 1:
        if yr_list[iyr] == start_yr:
            month_list = np.arange(start_m, 13)
        elif yr_list[iyr] == end_yr:
            month_list = np.arange(1,end_m+1)
        else:
            month_list = np.arange(1,13)
    else:
        month_list = np.arange(start_m, end_m+1)

    nmonths = len(month_list)
    
    for imonth in range(nmonths):
    
        c = cdsapi.Client()

        out_file = out_path + 'era5_1deg_ml137_'+str(yr_list[iyr])+"{:02d}".format(month_list[imonth])+'_download.grib'
        ndays = calendar.monthrange(yr_list[iyr], month_list[imonth])[1]
        date_str = str(yr_list[iyr])+'-'+"{:02d}".format(month_list[imonth])+'-01/to/'+str(yr_list[iyr])+'-'+"{:02d}".format(month_list[imonth])+'-'+"{:02d}".format(ndays)
    
        c.retrieve('reanalysis-era5-complete',
                   {'class': 'ea',
                    'date': date_str,
                    'expver': '1',
                    'levelist': '1/to/137',
                    'levtype': 'ml',
                    'param': '130/131/132/152',
                    'stream': 'oper',
                    'time': '00:00:00/to/21:00:00/by/03:00:00',
                    'type': 'an',
                    'grid' : '1.0/1.0'},
                   out_file)
    
        print(f'Downloaded file for {date_str}')

exit()
