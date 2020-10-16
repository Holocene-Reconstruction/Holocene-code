#==============================================================================
# A script to make some basic analyses of the Holocene Reconstruction.
#    author: Michael P. Erb
#    date  : 10/7/2020
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import xarray as xr

save_instead_of_plot = False


# Pick the reconstruction to analyze
recon_dir = '/projects/pd_lab/data/data_assimilation/results/'
recon_filename = 'holocene_recon_2020-09-30_18:47:56.491711_timevarying_5010yr_prior_gridded_output.nc'


### LOAD DATA

# Load the Holocene reconstruction
handle = xr.open_dataset(recon_dir+recon_filename,decode_times=False)
recon_gmt        = handle['recon_gmt'].values
prior_gmt        = handle['prior_gmt'].values
recon_mean       = handle['recon_mean'].values
proxy_values     = handle['proxy_values'].values
proxy_resolution = handle['proxy_resolutions'].values
proxy_metadata   = handle['proxy_metadata'].values
ages_da          = handle['ages'].values
lat              = handle['lat'].values
lon              = handle['lon'].values
options          = handle['options'].values
handle.close()

# Print the options used for this reconstruction
print('OPTIONS:')
for i in range(len(options)):
    name  = options[i].split(':')[0]
    value = options[i].split(':')[1]
    print('%-30s %-50s' % (name,value))


### CALCULATIONS

# Count data coverage
nproxies = proxy_values.shape[0]
data_counts = np.sum(np.isfinite(proxy_values),axis=0)


### FIGURES
plt.style.use('ggplot')

# Plot the main composite
plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((3,1),(0,0),rowspan=2)
ax2 = plt.subplot2grid((3,1),(2,0))

ax1.fill_between(ages_da,np.percentile(prior_gmt,2.5,axis=1),np.percentile(prior_gmt,97.5,axis=1),color='k',alpha=0.25)
ax1.fill_between(ages_da,np.percentile(recon_gmt,2.5,axis=1),np.percentile(recon_gmt,97.5,axis=1),color='b',alpha=0.5)
ax1.plot(ages_da,np.mean(prior_gmt,axis=1),'k',label='Prior')
ax1.plot(ages_da,np.mean(recon_gmt,axis=1),'b',label='Reconstruction')
ax1.legend()
ax1.set_xlim(ages_da[-1],ages_da[0])
ax1.set_ylabel('$\Delta$T ($^\circ$C)')
ax1.set_title('Global-mean temperature anomaly ($^\circ$C) from data assimilation\nusing calibrated proxies (n='+str(nproxies)+').  Mean and 95% range.',fontsize=18)

ax2.fill_between(ages_da,data_counts*0,data_counts,color='b',alpha=0.5)
ax2.set_title('Proxy data coverage')
ax2.set_xlim(ages_da[-1],ages_da[0])
ax2.set_ylim(ymin=0)
ax2.set_ylabel('# proxies')
ax2.set_xlabel('Age (B.P.)')

if save_instead_of_plot:
    plt.savefig('figure_gmt.png',dpi=150,format='png',bbox_inches='tight')
    plt.close()
else:
    plt.show()


# Function to make a map of a particular age in the reconstruction and all of the proxies
def map_recon_and_proxies(ages_anom,ages_ref):
    #
    # To help with plotting, make the longitudes run from -180 to 180
    indices_east = np.where(lon <= 180)[0]
    indices_west = np.where(lon > 180)[0]
    recon_mean_we = np.concatenate((recon_mean[:,:,indices_west],recon_mean[:,:,indices_east]),axis=2)
    lon_we = np.concatenate((lon[indices_west]-360,lon[indices_east]),axis=0)
    #
    # Compute the anomalies
    indices_anom = np.where((ages_da >= ages_anom[0]) & (ages_da <= ages_anom[1]))[0]
    indices_ref  = np.where((ages_da >= ages_ref[0])  & (ages_da <= ages_ref[1]))[0]
    recon_mean_we_for_age = np.nanmean(recon_mean_we[indices_anom,:,:],axis=0) - np.nanmean(recon_mean_we[indices_ref,:,:],axis=0)
    #
    # Get data for proxy
    proxy_values_for_age = np.nanmean(proxy_values[:,indices_anom],axis=1) - np.nanmean(proxy_values[:,indices_ref],axis=1)
    proxy_lats = proxy_metadata[:,2].astype(float)
    proxy_lons = proxy_metadata[:,3].astype(float)
    #
    # Make a map of changes, along with proxy values
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    lon_2d,lat_2d = np.meshgrid(lon_we,lat)
    x, y = m(lon_2d,lat_2d)
    x_proxy, y_proxy = m(proxy_lons,proxy_lats)
    #
    plt.figure(figsize=(14,9))
    anomaly_value = 2
    m.contourf(x,y,recon_mean_we_for_age,np.linspace(-1*anomaly_value,anomaly_value,21),extend='both',cmap='bwr',vmin=-1*anomaly_value,vmax=anomaly_value)
    m.drawcoastlines()
    m.colorbar(location='bottom')
    m.scatter(x_proxy,y_proxy,100,c=proxy_values_for_age,marker='o',edgecolor='k',alpha=1,cmap='bwr',vmin=-1*anomaly_value,vmax=anomaly_value,linewidths=1)
    plt.title('Temperature anomalies ($^\circ$C) for reconstruction and proxies\nat '+str(ages_anom[0])+'-'+str(ages_anom[1])+' vs. '+str(ages_ref[0])+'-'+str(ages_ref[1])+' yr BP',fontsize=18)
    #
    if save_instead_of_plot:
        plt.savefig('figure_map_age_'+str(ages_anom[0]).zfill(5)+'_'+str(ages_anom[1]).zfill(5)+'_vs_'+str(ages_ref[0]).zfill(5)+'_'+str(ages_ref[1]).zfill(5)+'_yr_BP.png',dpi=150,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Make a map of a particular age in the reconstruction and the assimilated proxies
map_recon_and_proxies([5500,6500],[0,1000])

