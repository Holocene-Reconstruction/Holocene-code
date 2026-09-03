#==============================================================================
# Downscaling the modeled precipitation. This is based on the method described
# in Lorenz et al., 2016. It starts with model output in a standard format with
# annual resolution.
#
# Method:
#  - Load model data and ERA5 reference data.
#  - Put both datasets in the same format, with years and months on different
#    axes.
#  - Compute spatial means of ERA5 to put it on the same grid as the model.
#  - Get model and reference data for the reference period.
#  - For every location and month, compare the model and reference values.
#  - Sort model data and reference data sorted by values at every location.
#  - For every location and month, use quantile mapping on the low resolution
#    data to translate model values for every age to ERA5 values using the
#    following rules:
#     - Values equal to 0: remain 0
#     - Values between 0 and the smallest value: interpolate between the
#       smallest value and 0
#     - Values greater than the largest value: use the slope of the entire
#       relationship to extend from the value of the last point.
#     - Values between points: Find the fraction between points and use that
#       when translating to the reference dataset
#  - On the newly mapped values, calculate each value as a fraction of the mean
#    at that location.
#  - Compute 100-year means (ideally, I would just do this at the end, but
#    doing it here makes the calculations faster/possible, and shouldn't impact
#    the results).
#  - Use bilinear interpolation and bias correct.
#  - Save the output.
#
# One addition: There are specific times and locations where the a normally-dry
# place in the model can have a relatively large value outside of the modern
# comparison period. This can produce exceptionally wet values in the quantile
# mapping. Values are constrained to be no larger that 2 times the ERA5 max.
#TODO: Is this a good approach?
#
# author: Michael Erb
#==============================================================================

import numpy as np
import xarray as xr
from scipy import stats
import copy

# Choose the model to downscale
model_to_downscale = "trace"
#model_to_downscale = "trace2"
#model_to_downscale = "itrace"
#model_to_downscale = "hadcm3"

model_dir = "P:/data_paleoclimate/data_assimilation/models/processed_model_data/"
era5_dir  = "P:/data_reanalyses/data_ERA5/"

filename_input  = model_to_downscale+".21999--100BP.precip.timeres_1.nc"
filename_output = model_to_downscale+"_downscaled.21999-0BP.precip.timeres_100.nc"

print(" === Processing model data. Downscaling, debiasing, and summing to 100 year resolution ===")
print("Input file:",filename_input)


#%% LOAD DATA

# Load the model data
xarray_model = xr.open_dataset(model_dir+filename_input)

# Trim the region, to save memory later
if max(xarray_model['lon'].values) > 270: xarray_model['lon'] = xarray_model['lon'] - 360
xarray_model = xarray_model.sel(lat=slice(-5,90),lon=slice(-180,0))

# Get model data
precip_model = xarray_model['precip'].values
lat_model    = xarray_model['lat'].values
lon_model    = xarray_model['lon'].values
age_model    = xarray_model['age'].values
xarray_model.close()
years_model = 1950 - age_model

# Load the ERA5 dataset
xarray_era5 = xr.open_dataset(era5_dir+'era5_monthly_tp_NorthAmerica.nc')
xarray_era5 = xarray_era5.sel(valid_time=slice('1940-01-01','2025-12-31'))  # Select full years
precip_era5 = xarray_era5['tp'].values
lat_era5    = xarray_era5['latitude'].values
lon_era5    = xarray_era5['longitude'].values
time_era5   = xarray_era5['valid_time'].values
xarray_era5.close()
years_era5_all = time_era5.astype('datetime64[Y]').astype(int) + 1970

# Convert units
precip_era5 = precip_era5 * 1000  # m/day to mm/day

# Get overview
print('ERA5, lats:', min(lat_era5), '-',max(lat_era5))
print('model, lats:',min(lat_model),'-',max(lat_model))
print('ERA5, lons:', min(lon_era5), '-',max(lon_era5))
print('model, lons:',min(lon_model),'-',max(lon_model))


#%% STEP 1: RESHAPE THE ERA5 DATA TO PUT YEARS AND MONTHS ON DIFFERENT AXES

# Reshape data and compute ages - era5
precip_era5_reshape = np.reshape(precip_era5,(int(precip_era5.shape[0]/12),12,len(lat_era5),len(lon_era5)))
years_era5 = np.arange(min(years_era5_all),max(years_era5_all)+1)
age_era5 = years_era5 - 1950


#%% STEP 2: GET VALUES FOR THE REFERENCE PERIOD

# Find the common period in the model and ERA5
year_min = max([min(years_model),min(years_era5)])
year_max = min([max(years_model),max(years_era5)])
years_ref = np.arange(year_min,year_max+1)

# Get precipitation values during the reference period
ind_ref_model = np.where((years_model >= year_min) & (years_model <= year_max))[0]
ind_ref_era5  = np.where((years_era5  >= year_min) & (years_era5  <= year_max))[0]
precip_model_ref = precip_model[ind_ref_model,:,:,:]
precip_era5_ref  = precip_era5_reshape[ind_ref_era5,:,:,:]


#%% STEP 3: BIN THE REFERENCE DATA TO THE SAME GRID AS THE MODEL

# Get the boundaries between gridcells
def get_boundaries(lat,lon):
    #
    lat_boundaries = copy.deepcopy(lat)
    lat_boundaries = np.insert(lat,0,lat[0]-(lat[1]-lat[0]))              # Add a point to the beginning
    lat_boundaries = np.append(lat_boundaries,lat[-1]+(lat[-1]-lat[-2]))  # Add a point to the end
    lat_boundaries = (lat_boundaries[:-1] + lat_boundaries[1:]) / 2       # Compute the means
    lat_boundaries[lat_boundaries >  90] =  90
    lat_boundaries[lat_boundaries < -90] = -90
    #
    lon_boundaries = copy.deepcopy(lon)
    lon_boundaries = np.insert(lon,0,lon[0]-(lon[1]-lon[0]))              # Add a point to the beginning
    lon_boundaries = np.append(lon_boundaries,lon[-1]+(lon[-1]-lon[-2]))  # Add a point to the end
    lon_boundaries = (lon_boundaries[:-1] + lon_boundaries[1:]) / 2       # Compute the means
    #
    print('Boundary ranges:')
    print('Latitude: ',lat_boundaries[0],'-',lat_boundaries[-1])
    print('Longitude:',lon_boundaries[0],'-',lon_boundaries[-1])
    #
    return lat_boundaries, lon_boundaries

# Get the boundaries between grid cells
lat_model_bounds,lon_model_bounds = get_boundaries(lat_model,lon_model)

# Set up low resolution variables
precip_era5_ref_lowres = np.zeros((len(years_ref),12,len(lat_model),len(lon_model))); precip_era5_ref_lowres[:] = np.nan

# Get weights
lat_weights = np.cos(np.radians(lat_era5))

# Compute spatial means
for j in range(len(lat_model)):
    print("Regridding ERA5 data ",j+1,"/",len(lat_model))
    for i in range(len(lon_model)):
        #
        j_selected = np.where((lat_era5 >= lat_model_bounds[j]) & (lat_era5 < lat_model_bounds[j+1]))[0]
        i_selected = np.where((lon_era5 >= lon_model_bounds[i]) & (lon_era5 < lon_model_bounds[i+1]))[0]
        if ((len(j_selected) > 0) & (len(i_selected) > 0)):
            #
            #print('Computing spatial mean. lats='+str(lat_era5[j_selected[0]])+'-'+str(lat_era5[j_selected[-1]])+', lons='+str(lon_era5[i_selected[0]])+'-'+str(lon_era5[i_selected[-1]])+'.  Points are inclusive.')
            precip_era5_ref_zonal = np.nanmean(precip_era5_ref[:,:,:,i_selected],axis=3)
            precip_era5_ref_lowres[:,:,j,i] = np.average(precip_era5_ref_zonal[:,:,j_selected],axis=2,weights=lat_weights[j_selected])


#%% STEP 4: SORT THE MODEL AND REFERENCE DATA

# For every location and month, make sure values are sorted from low to high
precip_model_ref_sorted       = np.sort(precip_model_ref,      axis=0)
precip_era5_ref_lowres_sorted = np.sort(precip_era5_ref_lowres,axis=0)
print(precip_model_ref_sorted.shape)
print(precip_era5_ref_lowres_sorted.shape)


#%% STEP 5: RESCALE PRECIPITATION VALUES FOR THE ENTIRE TIME SERIES

# Get dimensions
n_lat = len(lat_model)
n_lon = len(lon_model)

# Create an output variable
precip_model_adjusted = np.zeros((precip_model.shape)); precip_model_adjusted[:] = np.nan

#num_year = 14904
#num_month = 6
#num_lat = np.argmin(np.abs(lat_model - 31.54))
#num_lon = np.argmin(np.abs(lon_model - -112.5))

# Loop though all data points, adjusting them with the quantile mapping rules mentioned in the notes up top
num_year = 10; num_month = 5; num_lat = 1; num_lon = 4
for num_lon in range(n_lon):
    for num_lat in range(n_lat):
        print("Computing lon ",num_lon+1,"/",n_lon,", lat ",num_lat+1,"/",n_lat)
        for num_month in range(12):

            # Get values for location and month
            model_values = precip_model_ref_sorted[:,num_month,num_lat,num_lon]
            ref_values   = precip_era5_ref_lowres_sorted[:,num_month,num_lat,num_lon]
            model_min = model_values[0]
            model_max = model_values[-1]
            ref_min   = ref_values[0]
            ref_max   = ref_values[-1]
            
            # Compute slope at location
            slope,intercept,_,_,_ = stats.linregress(model_values,ref_values)  # slope,intercept,rvalue,pvalue,stderr
            
            # Loop through years at the month and location
            #for num_year in range(200):
            for num_year in range(precip_model.shape[0]):
                
                # Get values to correct
                value_original = precip_model[num_year,num_month,num_lat,num_lon]
                
                # Values equal to 0: remain 0
                if value_original == 0:
                    value_new = 0
                
                # Values between 0 and the smallest value: interpolate between the smallest point and 0
                elif value_original <= model_min:
                    value_new = (value_original / model_min) * ref_min
                
                # Values greater than the largest value: use the slope of the entire relationship to extend from the value of the last point.
                elif value_original >= model_max:
                    value_new = ref_max + (slope * (value_original - model_max))
                    if value_new > (2*ref_max): value_new = (2*ref_max)  # Constrain values to be no more than twice the reference maximum
                
                # Values between points: Find the fraction between points and use that when translating to the reference dataset
                else:
                    model_values_diff = value_original - model_values
                    model_values_diff[model_values_diff < 0] = np.nan
                    ind_closest_lower = np.nanargmin(model_values_diff)
                    ind_closest_upper = ind_closest_lower + 1
                    fraction_between = (value_original - model_values[ind_closest_lower]) / (model_values[ind_closest_upper] - model_values[ind_closest_lower])
                    value_new = ref_values[ind_closest_lower] + (fraction_between * (ref_values[ind_closest_upper] - ref_values[ind_closest_lower]))
                
                # Make a plot to test the results
                """
                import matplotlib.pyplot as plt
                plt.scatter(model_values,ref_values,c='tab:blue')
                #plt.scatter(np.mean(model_values),np.mean(ref_values),c='k')
                plt.scatter(value_original,value_new,c='tab:red')
                """
                # Set the new value
                precip_model_adjusted[num_year,num_month,num_lat,num_lon] = value_new


#%% STEP 6: CALCULATE VALUES AS A FRACTION OF THE CLIMATOLOGY OF THE REFERENCE PERIOD

#print(precip_model_adjusted.shape)
#print(precip_model_ref.shape)

precip_model_fraction = precip_model_adjusted / np.mean(precip_era5_ref_lowres,axis=0)[np.newaxis,:,:,:]


#%% STEP 7: COMPUTE 100-YEAR MEANS (TO MAKE A SMALLER FILE)

# Select years to keep and compute 100-year means
ind_selected = np.where((age_model < 22000) & (age_model >= 0))[0]
precip_model_fraction_100yr = np.nanmean(np.reshape(precip_model_fraction[ind_selected,:,:,:],(220,100,12,len(lat_model),len(lon_model))),axis=1)
age_100yr = np.nanmean(np.reshape(age_model[ind_selected],(220,100)),axis=1)
days_per_month_all = np.nanmean(np.reshape(xarray_model['days_per_month_all'][ind_selected,:].values,(220,100,12)),axis=1)

# Construct the variable to save
precip_fraction_100yr_df = xr.Dataset(
    {
        "precip":(["age","month","lat","lon"],precip_model_fraction_100yr),
        "days_per_month":    (["month"],      xarray_model['days_per_month'].values,{"units":"days"}),
        "days_per_month_all":(["age","month"],days_per_month_all,                   {"units":"days"}),
    },
    coords = {
        "age":   (["age"],  age_100yr,      {"units":"yr BP (ref 1950)"}),
        'month': (["month"],np.arange(1,13),{"units":"month_number"}),
        "lat":   (["lat"],  lat_model,      {"units":"degrees_north"}),
        "lon":   (["lon"],  lon_model,      {"units":"degrees_east"}),
    }
)


#%% STEP 8: USE BILINEAR INTERPOLATION TO DOWNSCALE TO THE REFERENCE DATASET

# Use bilinear interpolation to the resolution of ERA5
xarray_downscaled = precip_fraction_100yr_df.interp(lat=lat_era5,lon=lon_era5)
#xarray_downscaled.precip[0,0,:,:].plot()


#%% STEP 9: UN-NORMALIZE THE DATA USING THE CLIMATOLOGY OF THE REFERENCE DATASET

xarray_downscaled['precip'].values = xarray_downscaled['precip'].values * np.mean(precip_era5_ref,axis=0)[np.newaxis,:,:,:]
#xarray_downscaled['precip'].units = "mm/day"


#%% STEP 10: SAVE THE OUTPUT

xarray_downscaled.to_netcdf(model_dir+filename_output)
