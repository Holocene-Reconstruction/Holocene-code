#==============================================================================
# Downscaling for model results. This is based on the method described in
# Lorenz et al., 2016.
#
# Method for precipitation:
#  - Reformat the data to have years and months on different axes.
#  - Compute means of the reference dataset to put it on the same grid as the
#    model.
#  - Get model and reference data for the reference period.
#  - For every location and month, compare the model and reference values. Use
#    a scatterplot to visualize.
#  - Once we have a dataset of the model data and the reference data sorted by
#    values, we use that as our yardstick to translate other values, using the
#    following rules:
#     - Values equal to 0: remain 0
#     - Values between 0 and the smallest value: interpolate between the
#       smallest point and 0
#     - Values greater than the largest value: use the slope of the entire
#       relationship to extend from the value of the last point.
#     - Values between points: Find the fraction between points and use that
#       when translating to the reference dataset
#  - Do quantile mapping on the low resolution data
#  - Compute 100-year means (ideally, I would just do this at the end, but
#    doing it here makes the calculations faster/possible, and shouldn't impact
#    the results).
#  - Use bilinear interpolation and bias correct.
#  - Save the output.
#
#    author: Michael Erb
#==============================================================================

import sys
sys.path.append('C:/Users/erbm/Dropbox/Academia/NAU/Project_EcoClimate_Sensitivity/analysis/utils/')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import utils_ecoclimate as utils
from scipy import stats

save_regridded_data = False


#%% LOAD DATA

data_dir = "P:/data_models/trace21k_downscaling/"

# Load the TraCE-21ka data
xarray_trace = xr.open_mfdataset([data_dir+"trace_precip/trace.21999-0BP.cam2.h0.PRECC.nc",
                                  data_dir+"trace_precip/trace.21999-0BP.cam2.h0.PRECL.nc"],decode_times=False)

# Get trace data
precip_trace = xarray_trace['PRECC'].values + xarray_trace['PRECL'].values
lat_trace    = xarray_trace['lat'].values
lon_trace    = xarray_trace['lon'].values
time_trace   = xarray_trace['time'].values
xarray_trace.close()

# Load the ERA5 dataset
xarray_era5 = xr.open_dataset('P:/data_reanalyses/data_ERA5/era5_monthly_tp_NorthAmerica.nc')
precip_era5 = xarray_era5['tp'].values[:-4,:,:]  # Remove the last 4 months, since the file ends in April
lat_era5    = xarray_era5['latitude'].values
lon_era5    = xarray_era5['longitude'].values
#time_era5   = xarray_era5['valid_time'].values[:-4]
xarray_era5.close()

# Convert units
precip_trace = precip_trace * 60*60*24*1000  # m/s to mm/day
precip_era5  = precip_era5  * 1000           # m/day to mm/day

# Get overview
print('ERA5, lats:', min(lat_era5), '-',max(lat_era5))
print('trace, lats:',min(lat_trace),'-',max(lat_trace))
print('ERA5, lons:', min(lon_era5), '-',max(lon_era5))
print('trace, lons:',min(lon_trace),'-',max(lon_trace))


#%% STEP 1: RESHAPE THE DATA

# Reshape data and compute ages - trace
n_years_trace = int(len(time_trace)/12)
precip_trace_reshape = np.reshape(precip_trace,(n_years_trace,12,len(lat_trace),len(lon_trace)))
time_trace_reshape   = np.reshape(time_trace,  (n_years_trace,12))
age_trace_ann = np.mean(time_trace_reshape,axis=1)
age_trace_ann = np.ceil(age_trace_ann*1000)
years_trace_ann = age_trace_ann + 1950

# Reshape data and compute ages - era5
n_years_era5 = int(precip_era5.shape[0]/12)
precip_era5_reshape = np.reshape(precip_era5,(n_years_era5,12,len(lat_era5),len(lon_era5)))
years_era5_ann = np.arange(1940,2026)
age_era5_ann = years_era5_ann - 1950


#%% SAVE 100-YEAR MEANS (FOR REFERENCE)

# Select years to keep and compute 100-year means
ind_selected = np.where((age_trace_ann > -22000) & (age_trace_ann <= 0))[0]
precip_trace_100yr = np.nanmean(np.reshape(precip_trace_reshape[ind_selected,:,:,:],(220,100,12,len(lat_trace),len(lon_trace))),axis=1)
age_100yr = -1*np.nanmean(np.reshape(age_trace_ann[ind_selected],(220,100)),axis=1)

# Some additional information to save
ndays_per_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])  # TraCE-21k has a noleap calendar
ndays_per_month_all = np.repeat(ndays_per_month[np.newaxis,:],len(age_100yr),axis=0)

# Construct the variable to save
precip_100yr_df = xr.Dataset(
    {
        "precip":(["age","month","lat","lon"],precip_trace_100yr,{"units":"mm/day"}),
        "days_per_month":    (["month"],      ndays_per_month,    {"units":"days"}),
        "days_per_month_all":(["age","month"],ndays_per_month_all,{"units":"days"}),
    },
    coords = {
        "age":   (["age"],  age_100yr,      {"units":"yr BP (ref 1950)"}),
        'month': (["month"],np.arange(1,13),{"units":"month_number"}),
        "lat":   (["lat"],  lat_trace,      {"units":"degrees_north"}),
        "lon":   (["lon"],  lon_trace,      {"units":"degrees_east"}),
    }
)

precip_100yr_df.to_netcdf(data_dir+"trace.21999-0BP.precip.timeres_100.nc")


#%% STEP 2: GET VALUES FOR THE REFERENCE PERIOD

years_ref = np.arange(1940,1991)

ind_ref_trace = np.where((years_trace_ann >= years_ref[0]) & (years_trace_ann <= years_ref[-1]))[0]
precip_trace_ref = precip_trace_reshape[ind_ref_trace,:,:,:]

ind_ref_era5 = np.where((years_era5_ann >= years_ref[0]) & (years_era5_ann <= years_ref[-1]))[0]
precip_era5_ref = precip_era5_reshape[ind_ref_era5,:,:,:]


#%% STEP 3: BIN THE REFERENCE DATA TO THE SAME GRID AS THE MODEL

# Set up low resolution variables
precip_era5_ref_lowres = np.zeros((len(years_ref),12,len(lat_trace),len(lon_trace))); precip_era5_ref_lowres[:] = np.nan
lat_trace_bounds,lon_trace_bounds = utils.get_boundaries(lat_trace,lon_trace)

# Get weights
lat_weights = np.cos(np.radians(lat_era5))

# Compute spatial means
for j in range(len(lat_trace)):
    for i in range(len(lon_trace)):
        print(j,i)
        #
        j_selected = np.where((lat_era5 >= lat_trace_bounds[j]) & (lat_era5 < lat_trace_bounds[j+1]))[0]
        i_selected = np.where((lon_era5 >= lon_trace_bounds[i]) & (lon_era5 < lon_trace_bounds[i+1]))[0]
        if ((len(j_selected) > 0) & (len(i_selected) > 0)):
            #
            #print('Computing spatial mean. lats='+str(lat_era5[j_selected[0]])+'-'+str(lat_era5[j_selected[-1]])+', lons='+str(lon_era5[i_selected[0]])+'-'+str(lon_era5[i_selected[-1]])+'.  Points are inclusive.')
            precip_era5_ref_zonal = np.nanmean(precip_era5_ref[:,:,:,i_selected],axis=3)
            precip_era5_ref_lowres[:,:,j,i] = np.average(precip_era5_ref_zonal[:,:,j_selected],axis=2,weights=lat_weights[j_selected])


#%% SAVE RESHAPED DATA TO DATAFRAME

print(precip_trace_ref.shape)
print(precip_era5_ref_lowres.shape)

if save_regridded_data:
    
    # Create new datasets
    xarray_trace_and_era5 = xr.Dataset(
        {
            'precip_trace': (['year','month','lat','lon'],precip_trace_ref,      {'units':'mm/day'}),
            'precip_era5':  (['year','month','lat','lon'],precip_era5_ref_lowres,{'units':'mm/day'}),
        },
        coords={
            'year':  (['year'], years_ref,      {'units':'year'}),
            'month': (['month'],np.arange(1,13),{'units':'month_number'}),
            'lat':   (['lat'],  lat_trace,      {'units':'degrees_north'}),
            'lon':   (['lon'],  lon_trace,      {'units':'degrees_east'}),
        },
        attrs={
            'description':'Monthly precipitation in TraCE-21ka and ERA5 for 1940-1990',
        },
    )
    
    # Save the output
    xarray_trace_and_era5.to_netcdf(data_dir+'trace_and_era5_1940-1990CE_monthly_precip.nc')


#%% STEP 4: SORT THE MODEL AND REFERENCE DATA AND SAVE TO A COMMON VARIABLE

# Combine the sorted values
n_years = len(years_ref)
n_lat   = len(lat_trace)
n_lon   = len(lon_trace)
sorted_model_and_ref = np.zeros((2,n_years,12,n_lat,n_lon)); sorted_model_and_ref[:] = np.nan
sorted_model_and_ref[0,:,:,:,:] = np.sort(precip_trace_ref,axis=0)
sorted_model_and_ref[1,:,:,:,:] = np.sort(precip_era5_ref_lowres,axis=0)


#%% STEP 5: RESCALE VALUES FOR THE ENTIRE TIME SERIES

# Create an output variable
precip_trace_adjusted = np.zeros((precip_trace_reshape.shape)); precip_trace_adjusted[:] = np.nan
n_years_all = precip_trace_reshape.shape[0]

# Loop though all data points, adjusting them with the quantile mapping rules mentioned in the notes up top
num_year = 10; num_month = 3; num_lat = 3; num_lon = 7
for num_lon in range(n_lon):
    for num_lat in range(n_lat):
        print("Computing lon ",num_lon+1,"/",n_lon,", lat ",num_lat+1,"/",n_lat)
        for num_month in range(12):

            # Get values for location and month
            sorted_model_and_ref_location = sorted_model_and_ref[:,:,num_month,num_lat,num_lon]
            model_values = sorted_model_and_ref_location[0,:]
            ref_values   = sorted_model_and_ref_location[1,:]
            model_min = model_values[0]
            model_max = model_values[-1]
            ref_min   = ref_values[0]
            ref_max   = ref_values[-1]
            
            # Compute slope at location
            slope,intercept,_,_,_ = stats.linregress(model_values,ref_values)  # slope,intercept,rvalue,pvalue,stderr            
            
            # Slope, as calculated in the Lorenz code (why does this work?)
            #model_values2 = model_values - np.mean(model_values)
            #ref_values2 = ref_values - np.mean(ref_values)
            #slope2 = sum(model_values2*ref_values2) / sum(model_values2*model_values2)
            
            # Loop through years at the month and location
            #for num_year in range(200):
            for num_year in range(n_years_all):
                
                # Get values to correct
                value_original = precip_trace_reshape[num_year,num_month,num_lat,num_lon]
                
                # Values equal to 0: remain 0
                if value_original == 0:
                    value_new = 0
                
                # Values between 0 and the smallest value: interpolate between the smallest point and 0
                elif value_original <= model_min:
                    value_new = (value_original / model_min) * ref_min
                
                # Values greater than the largest value: use the slope of the entire relationship to extend from the value of the last point.
                elif value_original >= model_max:
                    value_new = ref_max + (slope * (value_original - model_max))
                
                # Values between points: Find the fraction between points and use that when translating to the reference dataset
                else:
                    model_values_diff = value_original - model_values
                    model_values_diff[model_values_diff < 0] = np.nan
                    ind_closest_lower = np.nanargmin(model_values_diff)
                    ind_closest_upper = ind_closest_lower + 1
                    fraction_between = (value_original - model_values[ind_closest_lower]) / (model_values[ind_closest_upper] - model_values[ind_closest_lower])
                    value_new = ref_values[ind_closest_lower] + (fraction_between * (ref_values[ind_closest_upper] - ref_values[ind_closest_lower]))
                
                # Make a plot to test the results
                #plt.scatter(model_values,ref_values,c='tab:blue')
                #plt.scatter(np.mean(model_values),np.mean(ref_values),c='k')
                #plt.scatter(value_original,value_new,c='tab:red')
                
                # Set the new value
                precip_trace_adjusted[num_year,num_month,num_lat,num_lon] = value_new


#%% STEP 6: CALCULATE VALUES AS A FRACTION OF THE CLIMATOLOGY OF THE REFERENCE PERIOD

#print(precip_trace_adjusted.shape)
#print(precip_trace_ref.shape)

precip_trace_fraction = precip_trace_adjusted / np.mean(precip_era5_ref_lowres,axis=0)[np.newaxis,:,:,:]


#%% STEP 7: COMPUTE 100-YEAR MEANS (TO MAKE A SMALLER FILE)

# Select years to keep and compute 100-year means
ind_selected = np.where((age_trace_ann > -22000) & (age_trace_ann <= 0))[0]
precip_trace_fraction_100yr = np.nanmean(np.reshape(precip_trace_fraction[ind_selected,:,:,:],(220,100,12,len(lat_trace),len(lon_trace))),axis=1)
age_100yr = -1*np.nanmean(np.reshape(age_trace_ann[ind_selected],(220,100)),axis=1)

# Construct the variable to save
precip_fraction_100yr_df = xr.Dataset(
    {
        "precip":(["age","month","lat","lon"],precip_trace_fraction_100yr),
        "days_per_month":    (["month"],      ndays_per_month,    {"units":"days"}),
        "days_per_month_all":(["age","month"],ndays_per_month_all,{"units":"days"}),
    },
    coords = {
        "age":   (["age"],  age_100yr,      {"units":"yr BP (ref 1950)"}),
        'month': (["month"],np.arange(1,13),{"units":"month_number"}),
        "lat":   (["lat"],  lat_trace,      {"units":"degrees_north"}),
        "lon":   (["lon"],  lon_trace,      {"units":"degrees_east"}),
    }
)


#%% STEP 8: USE BILINEAR INTERPOLATION TO DOWNSCALE TO THE REFERENCE DATASET

# Use bilinear interpolation to the resolution of ERA5
xarray_downscaled = precip_fraction_100yr_df.interp(lat=lat_era5,lon=lon_era5)
#var_100yr_df_downscaled.TREFHT[0,0,:,:].plot()


#%% STEP 9: UN-NORMALIZE THE DATA USING THE CLIMATOLOGY OF THE REFERENCE DATASET

xarray_downscaled['precip'].values = xarray_downscaled['precip'].values * np.mean(precip_era5_ref,axis=0)[np.newaxis,:,:,:]
xarray_downscaled['precip'].units = "mm/day"


#%% STEP 10: SAVE THE OUTPUT

xarray_downscaled.to_netcdf(data_dir+"trace_downscaled.21999-0BP.precip.timeres_100.nc")
