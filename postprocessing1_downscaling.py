#==============================================================================
# This script loads a monthly reconstruction, then downscales the temperature
# and precipitation fields and saves it in a reformatted way. Temperature is
# downscaled with linear additive scaling, and precipitation is done with
# linear multiplicative scaling (since quantile mapping would be difficult to
# do with 100-yr mean values. More details about the downscaling methods can be
# found in the subscripts.
# author: Michael Erb
#==============================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

save_instead_of_plot = False

# Pick the reconstruction to analyze
recon_filename = 'holocene_recon_2026-10-20_09_45_06_161721_eco21ka_trace_monthly'

# Pick the variable to process
var_to_process = 'tas'
#var_to_process = 'precip'


#%% LOAD DATA - RECONSTRUCTION

print(" === Processing reconstruction: downscaling and debiasing ===")

# Load the Holocene reconstruction
recon_dir = "P:/data_paleoclimate/data_assimilation/results/"
xarray_da = xr.open_dataset(recon_dir+recon_filename+".nc",decode_times=False)

# Load annual values
recon_annual = xarray_da['recon_'+var_to_process+'_annual_mean'].values
ages_da      = xarray_da['ages'].values
lat_da       = xarray_da['lat'].values
lon_da       = xarray_da['lon'].values
years_da = 1950 - ages_da

# Load monthly values
n_ages = recon_annual.shape[0]
n_lat  = recon_annual.shape[1]
n_lon  = recon_annual.shape[2]
recon_monthly = np.zeros((n_ages,12,n_lat,n_lon)); recon_monthly[:] = np.nan
for i in range(12):
    recon_monthly[:,i,:,:] = xarray_da['recon_'+var_to_process+'_month_'+str(i+1).zfill(2)+'_mean'].values

xarray_da.close()


#%% LOAD DATA - ERA5

# Load the ERA5 data
era5_dir  = "P:/data_reanalyses/data_ERA5/"
if var_to_process == 'tas':
    xarray_era5  = xr.open_dataset(era5_dir+'era5_monthly_t2m_NorthAmerica.nc')
    xarray_era5['t2m'] = xarray_era5['t2m'] - 273.15  # K to C
    xarray_era5 = xarray_era5.rename_vars({"t2m":"var_to_use"})
elif var_to_process == 'precip':
    xarray_era5  = xr.open_dataset(era5_dir+'era5_monthly_tp_NorthAmerica.nc')
    xarray_era5['tp'] = xarray_era5['tp'] * 1000  # m/day to mm/day
    xarray_era5 = xarray_era5.rename_vars({"tp":"var_to_use"})

xarray_era5["longitude"] = xarray_era5["longitude"] + 360

print(" === Lat ranges ===")
print("DA:",min(lat_da),max(lat_da))
print("ERA5:",min(xarray_era5["latitude"].values),max(xarray_era5["latitude"].values))
print(" === Lon ranges ===")
print("DA:",min(lon_da),max(lon_da))
print("ERA5:",min(xarray_era5["longitude"].values),max(xarray_era5["longitude"].values))


#%% STEP 1: CREATE NEW DATAFRAME

# Construct the variable to save
xarray_da_reformatted = xr.Dataset(
    {
        "recon_annual": (["age","lat","lon"],        recon_annual),
        "recon_monthly":(["age","month","lat","lon"],recon_monthly),
    },
    coords = {
        "age":   (["age"],  ages_da,        {"units":"yr BP (ref 1950)"}),
        'month': (["month"],np.arange(1,13),{"units":"month_number"}),
        "lat":   (["lat"],  lat_da,         {"units":"degrees_north"}),
        "lon":   (["lon"],  lon_da,         {"units":"degrees_east"}),
    }
)


#%% STEP 2: COMPUTE MEANS OVER THE YEARS OF OVERLAP

# Print the years
time_era5 = xarray_era5['valid_time'].values
print('DA years:',years_da[:3],years_da[-3:])
print('ERA5 years:',time_era5[:3],time_era5[-3:])

# For overlap, let's use:
# DA: years 1851-1950 (age=49.5)
# ERA5: years 1940-1950

# Compute monthly means over the selected reference period
da_mean_ref           = xarray_da_reformatted.sel(age=slice(0,100)).mean('age')
era5_mean_ref_monthly = xarray_era5.sel(valid_time=slice('1940-01-01','1950-12-31')).groupby('valid_time.month').mean('valid_time')

# Compute annual means over the selected reference period - ERA5
xarray_era5_selected = xarray_era5.sel(valid_time=slice('1940-01-01','1950-12-31'))
month_lengths = xarray_era5_selected.valid_time.dt.days_in_month
weights_annual = (month_lengths / month_lengths.sum())
era5_mean_ref_annual = (xarray_era5_selected.var_to_use * weights_annual).sum(dim="valid_time")

#plt.contourf(da_mean_ref['recon_annual'])
#plt.contourf(era5_mean_ref_annual)

#print(da_mean_ref)
#print(era5_mean_ref_annual)


#%% STEP 3: DOWNSCALE AND BIAS CORRECT

# Consider instituting a scaling cap, like 10, as mentioned here: 

# Interpolate model to ERA5
xarray_da_downscaled   = xarray_da_reformatted.interp(lat=xarray_era5["latitude"].values,lon=xarray_era5["longitude"].values)
da_mean_ref_downscaled = da_mean_ref.interp(lat=xarray_era5["latitude"].values,lon=xarray_era5["longitude"].values)
#xarray_da_downscaled.recon_monthly[0,0,:,:].plot()

# Add the bias correction to each timestep
#if var_to_process == 'tas':
if var_to_process in ['tas','precip']:
    adjustment_annual  = era5_mean_ref_annual.values                - da_mean_ref_downscaled['recon_annual'].values
    adjustment_monthly = era5_mean_ref_monthly['var_to_use'].values - da_mean_ref_downscaled['recon_monthly'].values
    xarray_da_downscaled['recon_annual']  = xarray_da_downscaled['recon_annual']  + adjustment_annual[np.newaxis,:,:]
    xarray_da_downscaled['recon_monthly'] = xarray_da_downscaled['recon_monthly'] + adjustment_monthly[np.newaxis,:,:,:]
elif var_to_process == 'precip':
    #TODO: This isn't working because the values are relative to a reference period rather than being absolute.
    adjustment_annual  = era5_mean_ref_annual.values                / da_mean_ref_downscaled['recon_annual'].values
    adjustment_monthly = era5_mean_ref_monthly['var_to_use'].values / da_mean_ref_downscaled['recon_monthly'].values
    # Set a max scaling factor of 10 (e.g., https://python-cmethods.readthedocs.io/en/v2.2.5/methods.html)
    adjustment_annual  = adjustment_annual.where((adjustment_annual   < 10) | (adjustment_annual.isnull()),  10)
    adjustment_monthly = adjustment_monthly.where((adjustment_monthly < 10) | (adjustment_monthly.isnull()), 10)
    xarray_da_downscaled['recon_annual']  = xarray_da_downscaled['recon_annual']  * adjustment_annual[np.newaxis,:,:]
    xarray_da_downscaled['recon_monthly'] = xarray_da_downscaled['recon_monthly'] * adjustment_monthly[np.newaxis,:,:,:]

# Make sure no precipitation values are below 0
if var_to_process == 'precip':
    #xarray_da_downscaled['recon_annual'].min()
    xarray_da_downscaled['recon_annual']  = xarray_da_downscaled['recon_annual'].where((xarray_da_downscaled['recon_annual']   > 0) | (xarray_da_downscaled['recon_annual'].isnull()),  0)
    xarray_da_downscaled['recon_monthly'] = xarray_da_downscaled['recon_monthly'].where((xarray_da_downscaled['recon_monthly'] > 0) | (xarray_da_downscaled['recon_monthly'].isnull()), 0)

"""
plt.contourf(era5_mean_ref_annual.values,levels=np.arange(0,10.1,.1))
plt.colorbar()
plt.show()

plt.contourf(da_mean_ref_downscaled['recon_annual'].values) # ,levels=np.arange(0,10.1,.1)
plt.colorbar()
plt.show()

plt.contourf(adjustment_annual,levels=np.arange(-10,10.1,.1))
"""


#%% MAKE MAPS TO CHECK

# Make maps
f, ax = plt.subplots(2,2,figsize=(16,10))
ax = ax.ravel()

da_mean_ref.recon_annual.plot(ax=ax[0])
era5_mean_ref_annual.plot(ax=ax[1])
xarray_da_reformatted.recon_annual[-1,:,:].plot(ax=ax[2])
xarray_da_downscaled.recon_annual[-1,:,:].plot(ax=ax[3])
if save_instead_of_plot:
    plt.savefig('figures/downscaling_check_'+var_to_process+'.png',dpi=150,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%% STEP 4: SAVE THE OUTPUT

# Change variable names
xarray_da_downscaled.rename_vars({"recon_annual":var_to_process+"_annual",
                                  "recon_monthly":var_to_process+"_monthly"})

# Save the data
xarray_da_downscaled.to_netcdf(recon_dir+recon_filename+"_downscaled_"+var_to_process+".nc")
