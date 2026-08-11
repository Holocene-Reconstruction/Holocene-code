#==============================================================================
# Downscaling for model results. This is based on the method described in
# Lorenz et al., 2016.
# 
# Method for temperature:
#  - Reformat the data to have years and months on different axes.
#  - Remove a reference period from the modeled output, to get anomalies.
#  - Compute 100-year means, since this is the final resolution I want, and
#    otherwise the files are too large.
#  - Use bilinear interpolation to put the model on the same resolution as the
#    reference dataset (ERA5)
#  - Add the modeled anomalies to the reference dataset, which has been
#    averaged over the reference period.
#  - Save the output.
#
#    author: Michael Erb
#==============================================================================

import numpy as np
import xarray as xr

# Choose the model to downscale
model_to_downscale = "trace"
#model_to_downscale = "trace2"
#model_to_downscale = "itrace"
#model_to_downscale = "hadcm3"


#%% LOAD DATA

# Load the model data
model_dir = "C:/Users/erbm/Documents/data_climate/data_assimilation/models/processed_model_data/"
xarray_model = xr.open_dataset(model_dir+model_to_downscale+".21999-0BP.tas.timeres_100.nc")  # decode_times=False

# Change the longitude from 0-360 to -360-0 and trim the region, to save memory later
if max(xarray_model['lon'].values) > 270: xarray_model['lon'] = xarray_model['lon'] - 360
xarray_model = xarray_model.sel(lat=slice(-5,90),lon=slice(-180,0))

# Get model data
var_model = xarray_model['tas'].values
lat_model = xarray_model['lat'].values
lon_model = xarray_model['lon'].values
age_model = xarray_model['age'].values
ndays_per_month = xarray_model['days_per_month'].values
xarray_model.close()
years_model = 1950 - age_model  #TODO: Check this

# Load the ERA5 dataset
xarray_era5 = xr.open_dataset('P:/data_reanalyses/data_ERA5/era5_monthly_t2m_NorthAmerica.nc')

# Convert units
var_model          = var_model          - 273.15  # K to C
xarray_era5['t2m'] = xarray_era5['t2m'] - 273.15  # K to C


#%% STEP 1: RESHAPE THE DATA

# Reshape data and compute ages
#n_years_model = int(len(time_model)/12)
#var_model_reshape  = np.reshape(var_model, (n_years_model,12,len(lat_model),len(lon_model)))
#time_model_reshape = np.reshape(time_model,(n_years_model,12))
#age_model_ann = np.mean(time_model_reshape,axis=1)
#age_model_ann = np.ceil(age_model_ann*1000)



#TODO: Pick up here.



#%% FIGURE

import sys
sys.path.append('C:/Users/erbm/Dropbox/Academia/NAU/Project_EcoClimate_Sensitivity/analysis/utils/')
import utils_ecoclimate as utils
import matplotlib.pyplot as plt

# Compute annual means
var_model_ann = np.average(var_model,axis=1,weights=ndays_per_month)

# Compute North American means for each month
region_na = [10,80,-173,-48]
var_model_ann_na = utils.spatial_mean(var_model_ann,lat_model,lon_model,region_na[0],region_na[1],region_na[2],region_na[3],lon_axis=2,lat_axis=1)

# Take a look at the recent period
plt.figure(figsize=(10,5))
ax1 = plt.subplot2grid((1,1),(0,0))
ax1.plot(years_model,var_model_ann_na)
ax1.set_xlim(1500,2000)
ax1.set_ylim(6,9)
ax1.set_title('Annual mean temperature for North American region',fontsize=18)
plt.show()


#%% STEP 4: COMPUTE ANOMALIES RELATIVE TO THE RECENT PERIOD

ref_years = [1950,1990]  # This is the period of overlap with ERA5
ind_ref = np.where((years_model_ann >= ref_years[0]) & (years_model_ann <= ref_years[1]))[0]
var_model_reshape = var_model_reshape - np.mean(var_model_reshape[ind_ref,:,:,:],axis=0)[np.newaxis,:,:,:]


#%% STEP 5: COMPUTE 100-YEAR MEANS (TO MAKE A SMALLER FILE)

# Select years to keep
ind_selected = np.where((age_model_ann > -22000) & (age_model_ann <= 0))[0]
var_model_reshape = var_model_reshape[ind_selected,:,:,:]
age_model_ann = age_model_ann[ind_selected]

# Compute 100-year means
var_100yr = np.nanmean(np.reshape(var_model_reshape,(220,100,12,len(lat_model),len(lon_model))),axis=1)
age_100yr = -1*np.nanmean(np.reshape(age_model_ann,(220,100)),axis=1)


#%% STEP 6: REFORMAT INTO A DATAFRAME

# Construct the variable to save
var_100yr_df = xr.Dataset(
    {
        "tas":(["age","month","lat","lon"],var_100yr,{"units":"degC"}),
        "days_per_month":    (["month"],      ndays_per_month,    {"units":"days"}),
        "days_per_month_all":(["age","month"],ndays_per_month_all,{"units":"days"}),
        },
    coords = {
        "age":   (["age"],  age_100yr,    {"units":"yr BP (ref 1950)"}),
        'month': (["month"],np.arange(1,13),{"units":"month_number"}),
        "lat":   (["lat"],  lat_model,    {"units":"degrees_north"}),
        "lon":   (["lon"],  lon_model,    {"units":"degrees_east"}),
    }
)


#%% STEP 7: DOWNSCALE AND BIAS CORRECT

# For the reference dataset, compute a mean over the common period
xarray_era5_common_period = xarray_era5.sel(valid_time=slice('1940-01-01','1990-12-31'))
xarray_era5_means         = xarray_era5_common_period.groupby('valid_time.month').mean('valid_time')
#xarray_era5_means.t2m[0,:,:].plot()

# Interpolate model to ERA5
lat_era5 = xarray_era5_means["latitude"].values
lon_era5 = xarray_era5_means["longitude"].values
var_100yr_df_downscaled = var_100yr_df.interp(lat=lat_era5,lon=lon_era5)
#var_100yr_df_downscaled.tas[0,0,:,:].plot()

# Add the bias correction to each timestep
var_100yr_df_downscaled['tas'] = var_100yr_df_downscaled['tas'] + xarray_era5_means['t2m'].values[np.newaxis,:,:,:]


#%% STEP 8: SAVE THE OUTPUT

# Save the data
#var_100yr_df_downscaled.to_netcdf("P:/data_models/trace21k_downscaling/"+model_to_downscale+"_downscaled.21999-0BP.tas.timeres_100.nc")
