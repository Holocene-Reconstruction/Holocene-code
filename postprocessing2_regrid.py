#==============================================================================
# This script regrids the reconstruction onto an equal area grid.
# author: Michael Erb
#==============================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xesmf as xe

plt.style.use('ggplot')
save_instead_of_plot = False


#%% LOAD DATA - RECONSTRUCTION

# Directories
recon_dir = "P:/data_paleoclimate/data_assimilation/results/"
grid_dir  = "C:/Users/erbm/Documents/GitHub/Holocene-code/data/"

# Load the Holocene reconstruction
filename_recon = 'holocene_recon_2026-10-20_09_45_06_161721_eco21ka_trace_monthly.nc'
xarray_da = xr.open_dataset(recon_dir+filename_recon,decode_times=False)
tas_da = xarray_da['recon_tas_annual_mean'].values
lat_da = xarray_da["lat"].values
lon_da = xarray_da["lon"].values
lon_da_2d,lat_da_2d = np.meshgrid(lon_da,lat_da)

# Load the downscaled Holocene reconstruction
filename_downscaled = 'holocene_recon_2026-10-20_09_45_06_161721_eco21ka_trace_monthly_downscaled_tas.nc'
xarray_downscaled = xr.open_dataset(recon_dir+filename_downscaled,decode_times=False)
lat_downscaled = xarray_downscaled["lat"].values
lon_downscaled = xarray_downscaled["lon"].values
lon_downscaled_2d,lat_downscaled_2d = np.meshgrid(lon_downscaled,lat_downscaled)

# Load the new grid
xarray_grid = xr.open_dataset(grid_dir+"US_Canada_NorthAmerica_Albers_50km_grid.nc",decode_times=False)
lat_grid              = xarray_grid["lat"].values
lon_grid              = xarray_grid["lon"].values
cell_id_grid          = xarray_grid["cell_id"].values
land_center_mask_grid = xarray_grid["land_center_mask"].values
crs_grid              = xarray_grid["crs"]

#plt.scatter(lon_grid,lat_grid,10,'k')


#%% MAP - LOOK AT NEW GRID

# Make a map
plt.figure(figsize=(20,10))
region_to_plot = [-195,-20,12,90]
region_center = (region_to_plot[0]+region_to_plot[1])/2
ax1 = plt.subplot2grid((2,2),(0,0),projection=ccrs.PlateCarree(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())

ax1.scatter(lon_grid,lat_grid,.5,'tab:blue',transform=ccrs.PlateCarree())
ax1.set_title('Equal area grid points',loc='center',fontsize=10)
ax1.coastlines()
ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
ax1.spines['geo'].set_edgecolor('black')

if save_instead_of_plot:
    plt.savefig('figures/grid_equal_area.png',dpi=300,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


#%% MAP - COMPARE GRIDS

# Make a map
plt.figure(figsize=(20,10))
region_to_plot = [-195,-20,10,90]
#region_to_plot = [-110,-100,30,40]
region_to_plot = [-125,-70,25,50]
region_center = (region_to_plot[0]+region_to_plot[1])/2
ax1 = plt.subplot2grid((2,2),(0,0),projection=ccrs.PlateCarree(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())
ax2 = plt.subplot2grid((2,2),(0,1),projection=ccrs.PlateCarree(central_longitude=region_center)); ax2.set_extent(region_to_plot,ccrs.PlateCarree())
ax3 = plt.subplot2grid((2,2),(1,0),projection=ccrs.PlateCarree(central_longitude=region_center)); ax3.set_extent(region_to_plot,ccrs.PlateCarree())

ax1.scatter(lon_da_2d,lat_da_2d,1,'tab:blue',transform=ccrs.PlateCarree())
ax1.set_title('DA grid - original',loc='center',fontsize=14)

ax2.scatter(lon_downscaled_2d,lat_downscaled_2d,1,'tab:blue',transform=ccrs.PlateCarree())
ax2.set_title('DA grid - downscaled',loc='center',fontsize=14)

ax3.scatter(lon_grid,lat_grid,1,'tab:blue',transform=ccrs.PlateCarree())
ax3.set_title('Equal area grid',loc='center',fontsize=14)

for ax in [ax1,ax2,ax3]:
    ax.coastlines()
    ax.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    #ax.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax.spines['geo'].set_edgecolor('black')

if save_instead_of_plot:
    plt.savefig('figures/grid_comparison.png',dpi=200,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()



#%% REGRID

regridder = xe.Regridder(xarray_da, xarray_grid, "bilinear")
xarray_da_regridded = regridder(xarray_da)
#print(list(xarray_da_regridded.keys()))
lat_new = xarray_da_regridded["y"].values
lon_new = xarray_da_regridded["x"].values

#%% MAP

# Make a map
plt.figure(figsize=(12,10))
region_to_plot = [-195,-20,10,90]
region_center = (region_to_plot[0]+region_to_plot[1])/2
ax1 = plt.subplot2grid((2,1),(0,0),projection=ccrs.PlateCarree(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())
ax2 = plt.subplot2grid((2,1),(1,0),projection=ccrs.PlateCarree(central_longitude=region_center)); ax2.set_extent(region_to_plot,ccrs.PlateCarree())

ax1.scatter(lon_da_2d,lat_da_2d,10,c=xarray_da['recon_tas_annual_mean'][-1,:,:].values,vmin=-10,vmax=10,marker='o',alpha=1,cmap='bwr',transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.set_title('Original DA',loc='center',fontsize=14)

ax2.scatter(lon_grid,lat_grid,10,c=xarray_da_regridded['recon_tas_annual_mean'][-1,:,:].values,vmin=-10,vmax=10,marker='o',alpha=1,cmap='bwr',transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.set_title('Regridded',loc='center',fontsize=14)

if save_instead_of_plot:
    plt.savefig('figures/equal_area_grid.png',dpi=200,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()



