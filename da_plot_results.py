#==============================================================================
# Make a map of the current state of the reconstruction
#    author: Michael Erb
#    date  : 11/6/2025
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature

plt.style.use('ggplot')

# Make a map of the current state of the reconstruction
def make_map(var_toplot,model_data,proxy_value,proxy_lat,proxy_lon,proxy_units,proxy_uncertainty,proxy,age,stage,exp_name_full,bounds=5,save_instead_of_plot=False):
    #
    # Make a map
    plt.figure(figsize=(10,12))
    region_to_plot = [-160,-40,0,80]
    region_center = (region_to_plot[0]+region_to_plot[1])/2
    ax1 = plt.subplot2grid((1,1),(0,0),projection=ccrs.LambertConformal(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())
    #
    recon_period_cyclic,lon_cyclic = cutil.add_cyclic_point(var_toplot,coord=model_data['lon'])
    map1 = ax1.contourf(lon_cyclic,model_data['lat'],recon_period_cyclic,np.linspace(-bounds,bounds,21),extend='both',cmap='bwr',transform=ccrs.PlateCarree())
    ax1.scatter(proxy_lon,proxy_lat,100,c=proxy_value,vmin=-bounds,vmax=bounds,marker='o',edgecolor='k',alpha=1,cmap='bwr',transform=ccrs.PlateCarree())
    colorbar1 = plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    colorbar1.ax.tick_params(labelsize=14)
    colorbar1.ax.set_facecolor('none')
    ax1.set_title('Proxy '+str(proxy)+': value='+str('%1.2f' % proxy_value)+' '+proxy_units+', uncertainty='+str('%1.2f' % proxy_uncertainty)+', '+stage,loc='center',fontsize=14)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax1.spines['geo'].set_edgecolor('black')
    #
    if save_instead_of_plot:
        plt.savefig('figures/map_da_'+exp_name_full+'_age_'+str(int(np.ceil(age))).zfill(5)+'yrBP_proxy_'+str(proxy).zfill(3)+'_'+stage+'.png',dpi=200,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
