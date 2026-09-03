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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import da_load_models
import da_utils_lmr
import copy

plt.style.use('ggplot')
save_instead_of_plot = True
output_dir = "C:/Users/erbm/Documents/GitHub/Holocene-code/sensitivity_experiments/figures/"

# Select time period of interest
age_intervals = np.linspace(0,22000,5)[:-1]
age_start = age_intervals[0]
ages_bounds = [age_start,age_start+5100]
#ages_bounds = [0,5100]
#ages_bounds = [8450,13550]
#ages_bounds = [16900,22000]


#%% LOAD AND PROCESS MODEL DATA

time_res = 100
options = {}
options['data_dir']            = 'P:/data_paleoclimate/data_assimilation/'
#options['models_for_prior']    = ['trace']; options['age_range_model'] = [0,22000]
options['models_for_prior']    = ['itrace']; options['age_range_model'] = [0,20000]
options['model_region']        = [0,85,180,350]
options['vars_to_reconstruct'] = ['tas_annual']
options['vars_root']           = ['tas']
#options['restrict_kalman']     = [True]
options['time_resolution']     = time_res
options['maximum_resolution']  = time_res
locrad_dist_km = 10000

# Load the chosen model data
model_data = da_load_models.load_model_data(options)


#%% CALCULATIONS

# Get model data
model_tas_annual = model_data['tas_annual']
model_lat = model_data['lat']
model_lon = model_data['lon']
model_age = model_data['age']

# Reshape model coordinates
n_latlon = len(model_lat) * len(model_lon)
lon_model_2d,lat_model_2d = np.meshgrid(model_lon,model_lat)
lat_prior = np.reshape(lat_model_2d,(n_latlon))
lon_prior = np.reshape(lon_model_2d,(n_latlon))
prior_coords = np.concatenate((lat_prior[:,None],lon_prior[:,None]),axis=1)

# Get data for the last 5100 years
ind_selected = np.where((model_age > ages_bounds[0]) & (model_age <= ages_bounds[1]))[0]
model_tas_annual = model_tas_annual[ind_selected,:,:]
model_tas_annual = model_tas_annual - np.nanmean(model_tas_annual,axis=0)
model_age = model_age[ind_selected]

# Set locations for covariance testing
lats_to_test = np.arange(20,70,20)
lons_to_test = np.arange(-120,-50,20) + 360
lats_to_test_all,lons_to_test_all = np.meshgrid(lats_to_test,lons_to_test)
lats_to_test_all = lats_to_test_all.flatten()
lons_to_test_all = lons_to_test_all.flatten()

# Set things up
n_locs = len(lats_to_test_all)
n_lat = len(model_lat)
n_lon = len(model_lon)
locs_all = np.zeros((n_locs,2)); locs_all[:] = np.nan
covariances_all  = np.zeros((n_locs,n_lat,n_lon)); covariances_all[:]  = np.nan
correlations_all = np.zeros((n_locs,n_lat,n_lon)); correlations_all[:] = np.nan
kmat_all         = np.zeros((n_locs,n_lat,n_lon)); kmat_all[:]         = np.nan
locrad_all       = np.zeros((n_locs,n_lat,n_lon)); locrad_all[:]       = np.nan

# Compute all covariance patterns
k=0
for k in range(n_locs):
    #
    print(k)
    #
    lat_selected = lats_to_test_all[k]
    lon_selected = lons_to_test_all[k]
    #
    # Get values near the selected location.
    j_selected = np.argmin(np.abs(lat_selected - model_lat))
    i_selected = np.argmin(np.abs(lon_selected - model_lon))
    locs_all[k,:] = model_lon[i_selected],model_lat[j_selected]
    #
    # Compute a localization radius filter
    locrad = da_utils_lmr.cov_localization(locrad_dist_km,lat_selected,lon_selected,prior_coords)
    locrad_2d = np.reshape(locrad,(n_lat,n_lon))
    locrad_all[k,:,:] = locrad_2d
    #
    # Get model values and compute covariances (The Kalman gain is cov/(var + error))
    model_values_selected = model_tas_annual[:,j_selected,i_selected]
    for j in range(n_lat):
        for i in range(n_lon):
            corr = np.corrcoef(model_values_selected,model_tas_annual[:,j,i])[0,1]
            cov  = np.cov(model_values_selected,model_tas_annual[:,j,i])[0,1]
            var  = np.var(model_values_selected)
            correlations_all[k,j,i] = corr
            covariances_all[k,j,i]  = cov
            kmat_all[k,j,i]         = cov/var

kmat_restricted_all = copy.deepcopy(kmat_all)
kmat_restricted_all[kmat_restricted_all < 0] = 0
kmat_restricted_all[kmat_restricted_all > 1] = 1


#%% FIGURES - MAPS

# A function to plot maps
value_txt,k,file_txt = "kalman_gain",1,'sample'
value_txt,k,file_txt = "kalman_gain_restricted",1,'sample'
value_txt,k,file_txt = "kalman_times_locrad",1,'sample'
value_txt,k,file_txt = "kalman_restricted_times_locrad",1,'sample'
def map_values(value_txt,k,file_txt):
    #
    #%%
    if   value_txt == "correlation":                    var_to_plot = correlations_all;               value_range = np.arange(-1,1.1,.1)
    elif value_txt == "covariance":                     var_to_plot = covariances_all;                value_range = np.arange(-.25,.251,.025)
    elif value_txt == "kalman_gain":                    var_to_plot = kmat_all;                       value_range = np.arange(-2,2.1,.2)
    elif value_txt == "kalman_gain_restricted":         var_to_plot = kmat_restricted_all;            value_range = np.arange(-2,2.1,.2)
    elif value_txt == "locrad":                         var_to_plot = locrad_all;                     value_range = np.arange(-1,1.1,.1)
    elif value_txt == "kalman_times_locrad":            var_to_plot = kmat_all*locrad_all;            value_range = np.arange(-2,2.1,.2)
    elif value_txt == "kalman_restricted_times_locrad": var_to_plot = kmat_restricted_all*locrad_all; value_range = np.arange(-2,2.1,.2)
    #
    # Make a map
    plt.figure(figsize=(10,12))
    region_to_plot = [-160,-40,0,80]
    region_center = (region_to_plot[0]+region_to_plot[1])/2
    ax1 = plt.subplot2grid((1,1),(0,0),projection=ccrs.LambertConformal(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())
    #
    map1 = ax1.contourf(model_lon,model_lat,var_to_plot[k,:,:],value_range,extend='both',cmap='bwr',transform=ccrs.PlateCarree())  # 
    ax1.scatter(locs_all[k,0],locs_all[k,1],100,c='yellow',marker='o',edgecolor='k',alpha=1,transform=ccrs.PlateCarree())
    colorbar1 = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    colorbar1.ax.tick_params(labelsize=14)
    colorbar1.ax.set_facecolor('none')
    ax1.set_title(value_txt+' for selected location\nwith everywhere else, TraCE-21k, '+str(ages_bounds[0])+'-'+str(ages_bounds[1])+' yr BP',loc='center',fontsize=18)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax1.spines['geo'].set_edgecolor('black')
    #
    if save_instead_of_plot:
        age_txt = 'ages_'+str(ages_bounds[0])+'_'+str(ages_bounds[1])
        plt.savefig(output_dir+'map1_'+options['models_for_prior'][0]+'_res_'+str(time_res)+'_loc_'+str(k).zfill(2)+'_'+age_txt+'_var'+file_txt+'_'+value_txt+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    #%%

for k in range(n_locs):
    #map_values("correlation",k,'1')
    #map_values("covariance",k,'2')
    #map_values("kalman_gain",k,'3')
    #map_values("kalman_gain_restricted",k,'3b')
    #map_values("locrad",k,'4')
    map_values("kalman_times_locrad",k,'5')
    #map_values("kalman_restricted_times_locrad",k,'5b')

