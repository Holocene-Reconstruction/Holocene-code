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
"""
var_toplot = var_toplot_updated
proxy = ind_proxy
stage = 'after'
save_instead_of_plot = False
"""
def make_map(var_toplot,model_data,proxy_value,proxy_lat,proxy_lon,proxy_units,proxy_uncertainty,proxy_psm,proxy,age,exp_name_full,save_instead_of_plot=False):
    #
    #%%
    # Make a map
    plt.figure(figsize=(12,10))
    region_to_plot = [-160,-40,0,80]
    region_center = (region_to_plot[0]+region_to_plot[1])/2
    ax1 = plt.subplot2grid((2,2),(0,0),projection=ccrs.LambertConformal(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())
    ax2 = plt.subplot2grid((2,2),(0,1),projection=ccrs.LambertConformal(central_longitude=region_center)); ax2.set_extent(region_to_plot,ccrs.PlateCarree())
    ax3 = plt.subplot2grid((2,2),(1,1),projection=ccrs.LambertConformal(central_longitude=region_center)); ax3.set_extent(region_to_plot,ccrs.PlateCarree())
    #
    map1 = ax1.contourf(model_data['lon'],model_data['lat'],var_toplot['tas_annual'],np.linspace(-2.5,2.5,21),extend='both',cmap='bwr',transform=ccrs.PlateCarree())
    plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    ax1.set_title('Temperature',loc='center',fontsize=14)

    map2 = ax2.contourf(model_data['lon'],model_data['lat'],var_toplot['precip_annual'],np.linspace(-.5,.5,21),extend='both',cmap='BrBG',transform=ccrs.PlateCarree())
    plt.colorbar(map2,orientation='horizontal',ax=ax2,fraction=0.08,pad=0.02)
    ax2.set_title('Precip',loc='center',fontsize=14)

    map3 = ax3.contourf(model_data['lon'],model_data['lat'],var_toplot['LakeStatus_annual'],np.linspace(-.5,.5,21),extend='both',cmap='BrBG',transform=ccrs.PlateCarree())
    plt.colorbar(map3,orientation='horizontal',ax=ax3,fraction=0.08,pad=0.02)
    ax3.set_title('Lake status',loc='center',fontsize=14)

    #print(proxy_units)
    if proxy_psm in ['calibrated_sst','calibrated_tas']:
        ax1.scatter(proxy_lon,proxy_lat,100,c=proxy_value,vmin=-2.5,vmax=2.5,marker='o',edgecolor='k',alpha=1,cmap='bwr',transform=ccrs.PlateCarree())
    elif proxy_psm == 'calibrated_precip':
        ax2.scatter(proxy_lon,proxy_lat,100,c=proxy_value,vmin=-.5,vmax=.5,marker='o',edgecolor='k',alpha=1,cmap='BrBG',transform=ccrs.PlateCarree())
    elif proxy_psm == 'rank_based_LakeStatus':
        ax3.scatter(proxy_lon,proxy_lat,100,c=proxy_value,vmin=-.5,vmax=.5,marker='o',edgecolor='k',alpha=1,cmap='BrBG',transform=ccrs.PlateCarree())

    for ax in [ax1,ax2,ax3]:
        ax.coastlines()
        ax.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
        ax.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
        ax.spines['geo'].set_edgecolor('black')
    #
    plt.suptitle('Proxy '+str(proxy)+': value='+str('%1.2f' % proxy_value)+' '+proxy_units+', uncertainty='+str('%1.2f' % proxy_uncertainty)+', '+proxy_psm,fontsize=14)  # +', '+stage
    plt.tight_layout()
    plt.subplots_adjust(top=.93)
    #
    if save_instead_of_plot:
        plt.savefig('figures/map2_multipanel_'+exp_name_full+'_age_'+str(int(np.ceil(age))).zfill(5)+'yrBP_proxy_'+str(proxy).zfill(3)+'.png',dpi=200,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()

#%% MAP - JUST FOR TEMPERATURE

def make_map_simple(var_toplot,model_data,proxy_value,proxy_lat,proxy_lon,proxy_units,proxy_uncertainty,proxy_psm,proxy,age,exp_name_full,save_instead_of_plot=False):
    #
    # Make a map
    plt.figure(figsize=(12,10))
    region_to_plot = [-160,-40,0,80]
    region_center = (region_to_plot[0]+region_to_plot[1])/2
    ax1 = plt.subplot2grid((1,1),(0,0),projection=ccrs.LambertConformal(central_longitude=region_center)); ax1.set_extent(region_to_plot,ccrs.PlateCarree())
    #
    map1 = ax1.contourf(model_data['lon'],model_data['lat'],var_toplot['tas_annual'],np.linspace(-2.5,2.5,21),extend='both',cmap='bwr',transform=ccrs.PlateCarree())
    plt.colorbar(map1,orientation='horizontal',ax=ax1,fraction=0.08,pad=0.02)
    ax1.set_title('Temperature',loc='center',fontsize=14)

    #print(proxy_units)
    if proxy_psm in ['calibrated_sst','calibrated_tas']:
        ax1.scatter(proxy_lon,proxy_lat,100,c=proxy_value,vmin=-2.5,vmax=2.5,marker='o',edgecolor='k',alpha=1,cmap='bwr',transform=ccrs.PlateCarree())
    else:
        ax1.scatter(proxy_lon,proxy_lat,50,c='k',marker='o',edgecolor='k',alpha=1,transform=ccrs.PlateCarree())

    ax1.coastlines()
    ax1.add_feature(cfeature.LAKES,facecolor='none',edgecolor='k')
    ax1.gridlines(color='k',linewidth=1,linestyle=(0,(1,5)))
    ax1.spines['geo'].set_edgecolor('black')
    ax1.set_title('Proxy '+str(proxy)+': value='+str('%1.2f' % proxy_value)+' '+proxy_units+', uncertainty='+str('%1.2f' % proxy_uncertainty)+', '+proxy_psm,fontsize=14)
    #
    if save_instead_of_plot:
        plt.savefig('figures/map1_simple_'+exp_name_full+'_age_'+str(int(np.ceil(age))).zfill(5)+'yrBP_proxy_'+str(proxy).zfill(3)+'.png',dpi=200,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
