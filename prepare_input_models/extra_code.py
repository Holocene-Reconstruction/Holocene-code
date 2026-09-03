
#%%

# Load the model data
xarray_model = xr.open_dataset(model_dir+filename_input)
xarray_era5  = xr.open_dataset('P:/data_reanalyses/data_ERA5/era5_monthly_tp_NorthAmerica.nc')

# Trim the region, to save memory later
if max(xarray_model['lon'].values) > 270: xarray_model['lon'] = xarray_model['lon'] - 360
#xarray_model_smaller = xarray_model.sel(lat=slice(30,50),lon=slice(-125,-75))

# Save smaller files
xarray_model_smaller = xarray_model.sel(lat=slice(35,45),lon=slice(-110,-90))
xarray_model_smaller.to_netcdf("C:/Users/erbm/Dropbox/Academia/"+model_to_downscale+".21999--100BP.precip.timeres_1.nc")

xarray_era5_smaller = xarray_era5.sel(latitude=slice(45,35),longitude=slice(-110,-90))
xarray_era5_smaller.to_netcdf("C:/Users/erbm/Dropbox/Academia/era5_monthly_tp_NorthAmerica.nc")


#%% LOAD DATA
