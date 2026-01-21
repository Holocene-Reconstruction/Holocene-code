#==============================================================================
# Load the multiple iterations from the same DA experiment and combine the
# results into a single file.
#    author: Michael Erb
#    date  : 1/14/2025
#==============================================================================

import sys
import numpy as np
import xarray as xr
import glob

#exp_txt = 'locrad_none'
#exp_txt = 'locrad_10k'
exp_txt = 'locrad_5k'
#exp_txt = 'locrad_5k_plus_rankbased'
#exp_txt = 'locrad_4k'
#exp_txt = 'locrad_3k'
#exp_txt = 'locrad_1k'
#exp_txt = sys.argv[1]

recon_dir = '/projects/pd_lab/data/data_assimilation/results/'
filenames_all = glob.glob(recon_dir+'*'+exp_txt+'_iter*')
filenames_all = np.sort(filenames_all)
n_iters = len(filenames_all)
print('Files found:',n_iters)


#%% LOAD AND CONCATENATE DATA

data_xarray_all = []

# Load the Holocene reconstruction
for i in range(n_iters):
    data_xarray = xr.open_dataset(filenames_all[i])
    data_xarray_all.append(data_xarray)

data_xarray_combined = xr.concat(data_xarray_all,dim='iters')

#%% LOAD THE FIRST FILE AND MAKE A FIRST DIMENTION TO HOLD ALL DATA



# Load the Holocene reconstruction
data_xarray_first = xr.open_dataset(filenames_all[0])
data_xarray_all = data_xarray_first.expand_dims(dim={'iter':10},axis=0)
var_list = list(data_xarray_first.keys())


#%% CHECK THE DIMENSIONS

for var_txt in var_list:
    print(var_txt,data_xarray_first[var_txt].shape,data_xarray_all[var_txt].shape)


#%% ADD DATA FROM OTHER FILES

file_num = 1
var_txt = 'recon_tas_mean'
for file_num in range(1,n_iters):
    print('Loading file',file_num)
    data_xarray_new = xr.open_dataset(filenames_all[file_num])
    for var_txt in var_list:
        n_dim = len(data_xarray_new[var_txt].shape)
        data_xarray_all[var_txt][file_num,:,:,:].values = data_xarray_new[var_txt].values  #TODO: Why isn't this working?
        print(data_xarray_all[var_txt][file_num,:3,:3,:3].values)
        print(data_xarray_new[var_txt][:3,:3,:3].values)
