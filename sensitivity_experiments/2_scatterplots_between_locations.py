#==============================================================================
# This script contains the main code of the Holocene data assimilation.
# Options are set in the config yml file. See README.txt for a more complete
# explanation of the code and setup.
# ---
# This version of the code has been updated to focus on North America for the
# North American ecological sensitivity project.
# ---
#    author: Michael Erb
#==============================================================================

# Change working directory
import os
os.chdir('C:/Users/erbm/Documents/GitHub/Holocene-code/')

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import da_load_models
import scipy

plt.style.use('ggplot')
save_instead_of_plot = False
output_dir = "C:/Users/erbm/Documents/GitHub/Holocene-code/sensitivity_experiments/figures/"


#%% LOAD AND PROCESS MODEL DATA

time_res = 100
options = {}
options['data_dir']            = 'P:/data_paleoclimate/data_assimilation/'
options['model_region']        = [0,85,180,350]
#options['models_for_prior']    = ['trace']; options['age_range_model'] = [0,22000]
options['models_for_prior']    = ['itrace']; options['age_range_model'] = [0,20000]
options['vars_to_reconstruct'] = ['tas_annual']
options['vars_root']           = ['tas']
options['time_resolution']     = time_res
options['maximum_resolution']  = time_res

# Load the chosen model data
model_data = da_load_models.load_model_data(options)

# Get model data
model_tas_annual = model_data['tas_annual']
model_lat = model_data['lat']
model_lon = model_data['lon']
model_age = model_data['age']


#%% CALCULATIONS

# Set locations to inspect
ages_bounds = [16900,22000]; loc1_latlon = [25,258]; loc2_latlon = [40,265]  # Strong negative correlations in trace
#ages_bounds = [11000,16100]; loc1_latlon = [40,240]; loc2_latlon = [45,260]  # Strong positive correlations in trace
#ages_bounds = [16900,22000]; loc1_latlon = [25,258]; loc2_latlon = [40,265]  # Mild negative correlations in trace

# Get values near the selected locations
j_selected1 = np.argmin(np.abs(loc1_latlon[0] - model_lat))
i_selected1 = np.argmin(np.abs(loc1_latlon[1] - model_lon))
tas_loc1 = model_tas_annual[:,j_selected1,i_selected1]
j_selected2 = np.argmin(np.abs(loc2_latlon[0] - model_lat))
i_selected2 = np.argmin(np.abs(loc2_latlon[1] - model_lon))
tas_loc2 = model_tas_annual[:,j_selected2,i_selected2]

# Get data for the selected period
ind_selected = np.where((model_age > ages_bounds[0]) & (model_age <= ages_bounds[1]))[0]

# Detrend both locations
model_age_selected = model_age[ind_selected]
tas_loc1_selected = tas_loc1[ind_selected]
tas_loc2_selected = tas_loc2[ind_selected]
slope_loc1,intercept_loc1,_,_,_ = scipy.stats.linregress(model_age_selected,tas_loc1_selected)  # slope,intercept,rvalue,pvalue,stderr
slope_loc2,intercept_loc2,_,_,_ = scipy.stats.linregress(model_age_selected,tas_loc2_selected)
tas_loc1_detrended = tas_loc1_selected - (model_age_selected * slope_loc1) + intercept_loc1
tas_loc2_detrended = tas_loc2_selected - (model_age_selected * slope_loc2) + intercept_loc2


#%% FIGURES

plt.figure(figsize=(20,6))
region_to_plot = [-120,-70,0,80]; region_center = (region_to_plot[0]+region_to_plot[1])/2
ax1_map = plt.subplot2grid((2,4),(0,0),rowspan=2,projection=ccrs.LambertConformal(central_longitude=region_center)); ax1_map.set_extent(region_to_plot,ccrs.PlateCarree())
ax2_ts = plt.subplot2grid((2,4),(0,1),colspan=2)
ax3_ts = plt.subplot2grid((2,4),(1,1),colspan=2)
ax4_scatter = plt.subplot2grid((2,4),(0,3),rowspan=2)

# Make a map
ax1_map.scatter(loc1_latlon[1],loc1_latlon[0],100,c='tab:blue',marker='o',alpha=1,transform=ccrs.PlateCarree())
ax1_map.scatter(loc2_latlon[1],loc2_latlon[0],100,c='tab:red', marker='o',alpha=1,transform=ccrs.PlateCarree())
ax1_map.set_title("Selected locations",loc='center',fontsize=14)
ax1_map.coastlines()
ax1_map.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
ax1_map.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
ax1_map.spines['geo'].set_edgecolor('black')

# Plot time series
ax2_ts.plot(model_age,tas_loc1,c='tab:blue')
ax3_ts.plot(model_age,tas_loc2,c='tab:red')
ax2_ts.set_title("Annual "+str(time_res)+"-yr temp at location 1",loc='center',fontsize=14)
ax3_ts.set_title("Annual "+str(time_res)+"-yr temp at location 2",loc='center',fontsize=14)
ax2_ts.set_xlim(22000,0)
ax3_ts.set_xlim(22000,0)
ax2_ts.axvspan(ages_bounds[0],ages_bounds[1],color="gray",alpha=0.25)
ax3_ts.axvspan(ages_bounds[0],ages_bounds[1],color="gray",alpha=0.25)

# Plot time series
corr = np.corrcoef(tas_loc1_selected,tas_loc2_selected)[0,1]
ax4_scatter.scatter(tas_loc1_selected,tas_loc2_selected)
#corr = np.corrcoef(tas_loc1_detrended,tas_loc2_detrended)[0,1]
#ax4_scatter.scatter(tas_loc1_detrended,tas_loc2_detrended)
ax4_scatter.set_title("Scatterplot, Corr = "+str("{:.2f}".format(corr)),loc='center',fontsize=14)
ax4_scatter.set_xlabel("Location 1 (blue)")
ax4_scatter.set_ylabel("Location 2 (red)")

if save_instead_of_plot:
    age_txt = 'ages_'+str(ages_bounds[0])+'_'+str(ages_bounds[1])
    plt.savefig(output_dir+'scatter_'+age_txt+'_locations.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()

