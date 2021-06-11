#==============================================================================
# This script takes a look at the eofs in simulation potentially used for the
# data assimilation prior.
#    author: Michael P. Erb
#    date  : 7/17/2020
#==============================================================================

import sys
sys.path.append('/home/mpe32/analysis/15_Holocene_Reconstruction/data_assimilation')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import da_utils
from scipy import signal
from scipy import stats

save_instead_of_plot = False


#%% LOAD DATA

time_resolution = 10

# Load the regridded HadCM3 model data
data_dir = '/projects/pd_lab/data/data_assimilation/models/processed_model_data/'
model_data = {}
handle_model = xr.open_dataset(data_dir+'hadcm3_regrid.12499-0BP.tas.timeres_'+str(time_resolution)+'.nc',decode_times=False)
model_data['tas']        = handle_model['tas'].values
model_data['lat']        = handle_model['lat'].values
model_data['lon']        = handle_model['lon'].values
model_data['age']        = handle_model['age'].values
model_data['time_ndays'] = handle_model['days_per_month_all'].values
handle_model.close()

# Compute annual means of the model data
n_lat = len(model_data['lat'])
n_lon = len(model_data['lon'])
time_ndays_model_latlon = np.repeat(np.repeat(model_data['time_ndays'][:,:,None,None],n_lat,axis=2),n_lon,axis=3)
model_data['tas_annual'] = np.average(model_data['tas'],axis=1,weights=time_ndays_model_latlon)


#%%
# A function filter a timeseries using a highpass or lowpass filter
#TODO: I think this function should work well, but look into it again before using it for anything important.
def filter_ts(data_ts,data_frequency,cutoff_frequency,pass_type,order=5):
    #
    # Inputs:
    # - data_ts:          The data time series
    # - data_frequency:   The frequency of the data time series.  Example: decadal=0.1 
    # - cutoff_frequency: The frequency of the desired cutoff.  Example: millennial=0.001 
    # - pass_type:        The type of filter.  Possibilities: high, low
    # - order:            This determines the order of the filter.  A higher number will match more variations.  I can adjust this as I see fit.
    #
    nyq = 0.5 * data_frequency
    normal_cutoff = cutoff_frequency / nyq
    b,a = signal.butter(order,normal_cutoff,btype=pass_type,analog=False)
    data_ts_filtered = signal.filtfilt(b,a,data_ts)
    #
    return data_ts_filtered


#%% CALCULATIONS

model_processing = 'linear_global'
#model_processing = 'linear_spatial'
#model_processing = 'highpass_global'
#model_processing = 'highpass_spatial'

#TODO: Check the lowpass and highpass filters.

# Get dimensions
n_time = len(model_data['age'])
n_lat  = len(model_data['lat'])
n_lon  = len(model_data['lon'])

# If desired, do a highpass filter on every location
if model_processing == 'linear_global':
    #
    #TODO: Should I handle individual months seperately?  Ask Luke what they did about this.
    tas_global = da_utils.global_mean(model_data['tas_annual'],model_data['lat'],1,2)
    slope,intercept,_,_,_ = stats.linregress(model_data['age'],tas_global)
    tas_global_linear = (model_data['age']*slope)+intercept
    tas_annual_filtered = model_data['tas_annual'] - tas_global_linear[:,None,None]
    tas_filtered        = model_data['tas']        - tas_global_linear[:,None,None,None]
    #
elif model_processing == 'linear_spatial':
    #
    tas_annual_filtered = np.zeros(model_data['tas_annual'].shape); tas_annual_filtered[:] = np.nan
    tas_filtered        = np.zeros(model_data['tas_annual'].shape); tas_filtered[:]        = np.nan
    for j in range(n_lat):
        for i in range(n_lon):
            slope,intercept,_,_,_ = stats.linregress(model_data['age'],model_data['tas_annual'][:,j,i])
            tas_linear = (model_data['age']*slope)+intercept
            tas_annual_filtered[:,j,i] = model_data['tas_annual'][:,j,i] - tas_linear
    #
elif model_processing == 'highpass_global':
    #
    data_frequency = 1/time_resolution
    cutoff_frequency = 0.001
    tas_global = da_utils.global_mean(model_data['tas_annual'],model_data['lat'],1,2)
    tas_global_lowpass = filter_ts(tas_global,data_frequency,cutoff_frequency,pass_type='low',order=5)
    tas_annual_filtered = model_data['tas_annual'] - tas_global_lowpass[:,None,None]
    #
elif model_processing == 'highpass_spatial':
    #
    data_frequency = 1/time_resolution
    cutoff_frequency = 0.001
    tas_annual_filtered = np.zeros(model_data['tas_annual'].shape); tas_annual_filtered[:] = np.nan
    for j in range(n_lat):
        for i in range(n_lon):
            tas_annual_filtered[:,j,i] = filter_ts(model_data['tas_annual'][:,j,i],data_frequency,cutoff_frequency,pass_type='high',order=5)


"""
plt.plot(model_data['age'],tas_global,'k-')
plt.plot(model_data['age'],model_data['age']*slope+intercept,'r-')
plt.show()
"""


#%% FIGURES
plt.style.use('ggplot')

#TODO: Why are these identical?  Are they?
tas_annual_filtered_global = da_utils.global_mean(tas_annual_filtered,model_data['lat'],1,2)

# Plot global-mean trends
plt.figure(figsize=(18,10))
ax1 = plt.subplot2grid((1,1),(0,0))

ax1.plot(model_data['age'],tas_global-np.mean(tas_global),'k-')
ax1.plot(model_data['age'],tas_annual_filtered_global-np.mean(tas_annual_filtered_global),'b-')
ax1.set_title('Global mean detrending, '+model_processing,fontsize=18)

if save_instead_of_plot:
    plt.savefig('figures/detrending_global_mean_'+model_processing+'.png',dpi=150,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()
