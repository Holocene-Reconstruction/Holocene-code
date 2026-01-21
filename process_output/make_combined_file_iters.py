#==============================================================================
# Load the multiple iterations from the same DA experiment and combine the
# results into a single file.
#    author: Michael Erb
#    date  : 1/15/2025
#==============================================================================

import sys
import numpy as np
import xarray as xr
import glob

exp_txt = sys.argv[1]


#%% LOAD DATA

recon_dir = '/projects/pd_lab/data/data_assimilation/results/'
filenames_all = glob.glob(recon_dir+'*'+exp_txt+'_iter*')
filenames_all = np.sort(filenames_all)
n_iters = len(filenames_all)
print('Processing '+exp_txt+'. files found:',n_iters)


#%% LOAD AND CONCATENATE DATA

# Load the data files
data_xarray_all = []
for i in range(n_iters):
    data_xarray = xr.open_dataset(filenames_all[i])
    data_xarray_all.append(data_xarray)

data_xarray_combined = xr.concat(data_xarray_all,dim='iters')


#%% SAVE OUTPUT

iter_first_txt = filenames_all[0].split('_')[-1].split('.')[0]
iter_last_txt  = filenames_all[-1].split('_')[-1].split('.')[0]
output_filename = '_'.join(filenames_all[-1].split('_')[:-2]) + '_combined_iters_'+iter_first_txt+'-'+iter_last_txt+'.nc'
data_xarray_combined.to_netcdf(output_filename)
