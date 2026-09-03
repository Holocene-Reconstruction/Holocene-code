#==============================================================================
# Make some maps of correlation patterns
#    author: Michael Erb
#==============================================================================

# Change working directory
import os
os.chdir('C:/Users/erbm/Documents/GitHub/Holocene-code/')

# Import libraries
import sys
sys.path.append('C:/Users/erbm/Dropbox/Academia/NAU/Project_EcoClimate_Sensitivity/analysis/utils/')
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import da_load_models
import utils_ecoclimate as utils

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


#%% CALCULATIONS

# Get model data
model_tas_annual = model_data['tas_annual']
model_lat = model_data['lat']
model_lon = model_data['lon']
model_age = model_data['age']

# Create a mask for North America
region_to_use = "continental"  # Just the main continental North America (slow)
#region_to_use = "all"          # Continental North America and all islands (very slow)
mask_na_trace = utils.mask_NorthAmerica(model_lat,model_lon,region_to_use)

# Compute the mean value for North America
tas_annual_namer  = utils.mean_of_selected(model_tas_annual,model_lat,model_lon,mask_na_trace)


#%% FIGURES - MAPS

# A function to plot maps
age_new,age_old = 0,5100
age_new,age_old = 17000,22000
def map_correlations(age_new,age_old,file_num):
    #
    # Get data for the selected period
    ind_selected = np.where((model_age > age_new) & (model_age <= age_old))[0]
    tas_period = model_tas_annual[ind_selected,:,:]
    tas_namer_period = tas_annual_namer[ind_selected]
    #
    # Set things up
    n_lat = len(model_lat)
    n_lon = len(model_lon)
    correlations_all = np.zeros((n_lat,n_lon)); correlations_all[:] = np.nan
    #
    # Compute correlations
    for j in range(n_lat):
        for i in range(n_lon):
            correlations_all[j,i] = np.corrcoef(tas_namer_period,tas_period[:,j,i])[0,1]
    #
    # Make a figures
    plt.figure(figsize=(10,14))
    region_to_plot = [-160,-40,0,80]; region_center = (region_to_plot[0]+region_to_plot[1])/2
    ax1 = plt.subplot2grid((4,1),(0,0),rowspan=3,projection=ccrs.LambertConformal(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())
    ax2 = plt.subplot2grid((4,1),(3,0))
    #
    # Make a map
    map1 = ax1.contourf(model_lon,model_lat,correlations_all,np.arange(-1,1.1,.1),extend='both',cmap='bwr',transform=ccrs.PlateCarree())  # 
    #ax1.scatter(locs_all[k,0],locs_all[k,1],100,c='yellow',marker='o',edgecolor='k',alpha=1,transform=ccrs.PlateCarree())
    colorbar1 = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    colorbar1.ax.tick_params(labelsize=14)
    colorbar1.ax.set_facecolor('none')
    ax1.set_title("Correlation between each location and the North American mean\nfor annual-mean "+str(time_res)+"-year temperature in TraCE-21k, "+str(age_new)+'-'+str(age_old)+' yr BP',loc='center',fontsize=14)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax1.spines['geo'].set_edgecolor('black')
    #
    # Plot time series
    ax2.plot(model_age,tas_annual_namer,c='tab:blue')
    ax2.set_title("Mean "+str(time_res)+"-yr temp averaged over North America",loc='center',fontsize=14)
    ax2.set_xlim(22000,0)
    ax2.axvspan(age_new,age_old,color="gray",alpha=0.25)
    #
    if save_instead_of_plot:
        age_txt = 'ages_'+str(age_new).zfill(5)+'_'+str(age_old).zfill(5)
        plt.savefig(output_dir+'map3_'+str(file_num).zfill(2)+'_res_'+str(time_res).zfill(3)+'_'+age_txt+'.png',dpi=300,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

age_old_all = np.arange(22100,5000,-1000)
for i,age_old in enumerate(age_old_all):
    age_new = age_old - 5100
    map_correlations(age_new,age_old,i)

