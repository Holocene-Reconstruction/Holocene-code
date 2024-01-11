import os
wd = '/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP21k/' 
os.chdir(wd+'Holocene-code') #changed
#Math functions
import pyet               # Package for calciulating potential ET
import xarray as xr
import numpy  as np
#Plotting Functions
import da_utils_plotting  as da_plot
import da_load_proxies
import da_utils_lakestatus as da_utils_ls
import matplotlib.pyplot  as plt      
import matplotlib.patches as patches
from matplotlib import cm
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes  
import cartopy.crs        as ccrs
import cartopy.feature    as cfeature

#Settings
font = {'family': 'sans-serif', 'sans-serif': 'Lucida Grande'}
geolims = [-125,-30,-47,40]
wd1 = '/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP21k/'
wd = wd1+'Data/models/LowryMorrillCCSM4/'
#proxy_data=proxy_data

#%%
#Lowry and Morrill (2019)
lm19 = {}
for age in ['21k','PI']:
    lm19[age]={}
    for reg in ['NAM','SAM']:
        #Load Data
        handle_model=xr.open_dataset(wd+age+'-CCSM4-'+reg+'-10m.nc', decode_times=False)
        vals = handle_model
        vals = vals.load()
        handle_model.close
        #
        lm19[age][reg]=vals   
#%%CCSM data
#https://www.earthsystemgrid.org/dataset/cmip5.output1.NCAR.CCSM4.lgm.mon.atmos.Amon.r1i1p1/file.html
#https://www.earthsystemgrid.org/dataset/cmip5.output1.NCAR.CCSM4.piControl.mon.atmos.Amon.r1i1p1/file.html
gcm = {}
for age in ['21k','PI']:
    gcm[age] = {}
    if age == '21k':  fn = '_Amon_CCSM4_lgm_r1i1p1_180001-190012.nc'
    elif age == 'PI': fn = '_Amon_CCSM4_piControl_r1i1p1_080001-130012.nc'
        #fn = '_Amon_CCSM4_piControl_r1i1p1_025001-050012.nc'
        #fn = '_Amon_CCSM4_piControl_r1i1p1_050101-079912.nc'
    #
    for var in ['tas','ps','hurs','pr','evspsbl','rlds','rlus','rsds','rsus','mrros']:
        if var == 'mrros': fn=fn.replace("Amon", "Lmon")
        model = xr.open_dataset(wd+'CCSM4/'+age+'/'+var+fn)
        gcm[age][var] = model[var]
        gcm[age][var]=gcm[age][var].load()
        gcm[age][var] = gcm[age][var].groupby('time.year').mean(dim='time')[0:100].rename({'year':'time'})
        gcm[age][var]['time'] = range(0,100)
        #elif age == '21k': gcm[age][var]['time'] = range(100,00)
        gcm[age][var]= gcm[age][var].assign_attrs({'units':        model[var].units,
                                                    'standard_name':model[var].standard_name,
                                                    'long_name':    model[var].long_name,
                                                    'cell_methods': model[var].cell_methods,
                                                    'comment':      model[var].comment
                                                    })
        model.close()
        #print(age+' - '+var+' - '+gcm[age][var].long_name+' - '+gcm[age][var].units)
    #print(np.shape(gcm[age][var]))
    #
    #Calculate Net Radiatio
    gcm[age]['SWnet'] = (gcm[age]['rsds'] - gcm[age]['rsus'])
    gcm[age]['SWnet'] = gcm[age]['SWnet'].assign_attrs({'units':gcm[age]['rsds'].units,'standard_name':'SW net radiation','comment':'rsds-rsus'})
    gcm[age]['LWnet'] = (gcm[age]['rlds'] - gcm[age]['rlus'])
    gcm[age]['LWnet'] = gcm[age]['SWnet'].assign_attrs({'units':gcm[age]['rlds'].units,'standard_name':'LW net radiation','comment':'rlds-rlus'})
    gcm[age]['radnet'] = (gcm[age]['rlds'] - gcm[age]['rlus'])
    gcm[age]['radnet'] = gcm[age]['SWnet'].assign_attrs({'units':gcm[age]['rsds'].units,'standard_name':'net radiation','comment':'(rsds-rsus)+(rlds-rlus)'})
    #
    #Combine lake evaporationMorrill and Li (2019) data onto same grid
    gcm[age]['LM19_lakeEvap'] =  (gcm[age]['evspsbl'] * np.NaN).rename('LEVAP')
    for reg in ['NAM','SAM']:
        lati = [i for i, element in enumerate(gcm[age]['LM19_lakeEvap'].lat.data) if element in lm19[age][reg].lat.data]
        loni = [i for i, element in enumerate(gcm[age]['LM19_lakeEvap'].lon.data) if element in lm19[age][reg].lon.data]
        gcm[age]['LM19_lakeEvap'][:,lati,loni]  = da_utils_ls.lm19_2_gcmFromat(lm19[age][reg]['LEVAP'])
        gcm[age]['LM19_lakeEvap'].where(gcm[age]['LM19_lakeEvap']>-99)
    gcm[age]['LM19_lakeEvap'] = gcm[age]['LM19_lakeEvap'].assign_attrs({'units':'cm/month','standard_name':'Lake Evaporation','comment':'Lake Evaporation values from Li and Morill (2019) model'})
    #
    #Combine wind speed Morrill and Li (2019) data onto same grid
    gcm[age]['LM19_U2'] =  (gcm[age]['evspsbl'] * np.NaN).rename('U2')
    for reg in ['NAM','SAM']:
        lati = [i for i, element in enumerate(gcm[age]['LM19_U2'].lat.data) if element in lm19[age][reg].lat.data]
        loni = [i for i, element in enumerate(gcm[age]['LM19_U2'].lon.data) if element in lm19[age][reg].lon.data]
        gcm[age]['LM19_U2'][:,lati,loni]  = da_utils_ls.lm19_2_gcmFromat(lm19[age][reg]['U2'])
        gcm[age]['LM19_U2'].where(gcm[age]['LM19_U2']>-99)
    gcm[age]['LM19_U2'] = gcm[age]['LM19_lakeEvap'].assign_attrs({'units':'m/s','standard_name':'wind speed','comment':'2-meter wind speed values from Li and Morill (2019) model'})
    #
    # Convert data as neeeded
    # K to degC
    gcm[age]['tas'].data = gcm[age]['tas'].data-273.15;
    gcm[age]['tas']=gcm[age]['tas'].assign_attrs({'units':'degC'})
    #Pa to kPa
    gcm[age]['ps'].data = gcm[age]['ps'].data*0.001; 
    gcm[age]['ps']=gcm[age]['ps'].assign_attrs({'units':'kPa'})
    # kg/m2/s to mm/day 
    gcm[age]['pr'].data = gcm[age]['pr'].data*(60*60*24);      
    gcm[age]['pr'] = gcm[age]['pr'].assign_attrs({'units':'mm/day'})
    # 
    gcm[age]['evspsbl'].data = gcm[age]['evspsbl'].data*(60*60*24);    
    gcm[age]['evspsbl'] = gcm[age]['evspsbl'].assign_attrs({'units':'mm/day'})
    #
    gcm[age]['mrros'].data = gcm[age]['mrros'].data*(60*60*24);    
    gcm[age]['mrros'] = gcm[age]['mrros'].assign_attrs({'units':'mm/day'})
    # cm/month to mm/day
    gcm[age]['LM19_lakeEvap'].data = gcm[age]['LM19_lakeEvap'].data*(10/30); 
    gcm[age]['LM19_lakeEvap']=gcm[age]['LM19_lakeEvap'].assign_attrs({'units':'mm/day'})
    # W/m2 to MJ/(m2*day)
    gcm[age]['radnet'].data = gcm[age]['radnet'].data*0.0864; 
    gcm[age]['radnet']=gcm[age]['radnet'].assign_attrs({'units':'MJ/(m2*day)'})
    # Adjust rounding error
    gcm[age]['mrros']['lat'] = gcm[age]['pr']['lat']
    gcm[age]['mrros']['lon'] = gcm[age]['pr']['lon']
    #Calculate PET
    gcm[age]['evspsbl'].data = np.where(gcm[age]['evspsbl']>0,gcm[age]['evspsbl'],np.abs(gcm[age]['evspsbl']).min(dim='time'))
    #gcm[age]['evspsbl'].data = gcm[age]['evspsbl'].data + (np.min(gcm[age]['evspsbl'].data,axis=0)+0.000001)
    #gcm[age]['evspsbl'].data = np.where(gcm[age]['evspsbl']>0,gcm[age]['evspsbl'],(np.min(gcm[age]['evspsbl'].data,axis=0)+0.000001))
    #gcm[age]['evspsbl'] = gcm[age]['evspsbl'].where(gcm[age]['evspsbl']>0)
    #
    gcm[age]['PETpenman'] = pyet.penman(tmean=    gcm[age]['tas'],    # degC
                                        wind=      gcm[age]['LM19_U2'],# m/s
                                        rn=        gcm[age]['radnet'], # MJ m-2 d-1 
                                        rh=        gcm[age]['hurs'],   # %
                                        pressure = gcm[age]['ps'])     # kPa
    gcm[age]['PETpr'] = pyet.priestley_taylor(tmean=    gcm[age]['tas'],    # degC
                                              #wind=      gcm[age]['LM19_U2'],# m/s
                                              rn=        gcm[age]['radnet'], # MJ m-2 d-1 
                                              rh=        gcm[age]['hurs'],   # %
                                              pressure = gcm[age]['ps'])     # kPa


#%%Load proxy data
import sys
import time
import yaml
import datetime

starttime_total = time.time() # Start timer

# Use a given config file.  If not given, use config_default.yml.
if len(sys.argv) > 1: config_file = sys.argv[1]
#else:                 config_file = 'config.yml'
else:                 config_file = 'config_default.yml'

# Load the configuration options and print them to the screen.
print('Using configuration file: '+config_file)
with open(config_file,'r') as file: options = yaml.load(file,Loader=yaml.FullLoader)

print('=== SETTINGS ===')
for key in options.keys():
    print('%30s: %-15s' % (key,str(options[key])))
print('=== END SETTINGS ===')


options['exp_name_long'] = options['exp_name']+'.'+str(options['localization_radius'])+'loc.'+str(options['prior_window'])+'window.'+str(options['time_resolution'])+'.'
# Load the chosen proxy data
proxy_ts,collection_all = da_load_proxies.load_proxies(options)
proxy_data = da_load_proxies.process_proxies(proxy_ts,collection_all,options)

#%% Which grid cells match proxy locations
proxyGrid = gcm[age]['LM19_lakeEvap'][0]*np.NaN
for i in range(len(proxy_data['lats'])):
    lat = proxy_data['lats'][i]
    lon = proxy_data['lons'][i]
    lati = np.argmin(np.abs(proxyGrid.lat.data-proxy_data['lats'][i]))
    loni = np.argmin(np.abs(proxyGrid.lon.data-proxy_data['lons'][i]))
    proxyGrid[lati,loni] = 1
#%%
gcm['combined']={}
for var in gcm['21k'].keys(): 
    gcm['21k'][var]['time'] = np.array([*range(100,200,1)])
    gcm['combined'][var] = xr.concat([gcm['PI'][var],gcm['21k'][var]],dim='time')
    #time_bins = range(0,201,5)
    #gcm['combined'][var] = gcm['combined'][var].groupby_bins('time', bins=time_bins).mean(dim='time').rename({'time_bins':'time'})

q = 'P/E'
age='combined'
gcm[age]['LakeStatusPET']   = da_utils_ls.calcLakeStatus(runoff=gcm[age]['mrros'],precip=gcm[age]['pr'],evap=gcm[age]['evspsbl'],levap=gcm[age]['PETpr'],Qmethod=q)
gcm[age]['LakeStatusLEVAP'] = da_utils_ls.calcLakeStatus(runoff=gcm[age]['mrros'],precip=gcm[age]['pr'],evap=gcm[age]['evspsbl'],levap=gcm[age]['LM19_lakeEvap'],Qmethod='runoff')
if q == 'P/E': qname = 'P:E'
else: qname == q
#%%    
fs=8
#Scatter plot of lake percentile
for age in ['combined']:
    dataX = gcm[age]['LakeStatusLEVAP'][:]*100
    dataY = gcm[age]['LakeStatusPET'][:]*100
    #
    time_bins = range(0,101,10)
    #dataX = dataX.groupby_bins('time', bins=time_bins).mean(dim='time').rename({'time_bins':'time'})
    #dataY = dataY.groupby_bins('time', bins=time_bins).mean(dim='time').rename({'time_bins':'time'})
    #
    #Set up Plot
    plt.figure(figsize=(3,2.5),dpi=400); plt.rc('font', **font)
    ax = plt.axes()
    #Plot
    ax.scatter(dataX,dataY,                                                            s=0.4, c= 'teal',  lw=0,alpha=0.01,label='All sites')
    #ax.scatter(dataX.where(np.isfinite(proxyGrid)),dataY.where(np.isfinite(proxyGrid)),s=1, c= 'purple',lw=0,alpha=0.1,label='Proxy sites')
    ax.axline((0,0),(30,30),linestyle='--',c='k',linewidth=0.5,label='1:1 line')
    ax.set_xlim((0,100)); ax.set_xticks(range(0,101,25)); 
    ax.set_ylim((0,100)); ax.set_yticks(range(0,101,25)); 
    #Add Skill Annotation
    ax.annotate('r\u00b2 = '+str(da_utils_ls.calcSkill(dataX,dataY,'r2',w=True)),#+ ' ('+str(calcSkill(dataX.where(np.isfinite(proxyGrid)),dataY.where(np.isfinite(proxyGrid)),'r2'))+')',
                (0.25,0.9),fontsize=fs,xycoords='axes fraction',ha='center')
    ax.annotate('RMSE = '+str(np.round(da_utils_ls.calcSkill(dataX,dataY,'RMSE',w=True),1)),#+ ' ('+str(np.round(calcSkill(dataX.where(np.isfinite(proxyGrid)),dataY.where(np.isfinite(proxyGrid)),'RMSE'),1))+')',
                (0.25,0.83),fontsize=fs-1,xycoords='axes fraction',ha='center')
    rectangle = patches.Rectangle((0.05, 0.80), 0.4, 0.18, transform=ax.transAxes, facecolor='w', edgecolor='black',lw=0.4)
    ax.add_patch(rectangle)
    #
    if q == 'runoff': 
        ax.set_title('Impact on lake status',fontsize=fs)
        ax.annotate('(b) ',(0.05,0.89),fontsize=fs,xycoords='figure fraction',ha='left')
    else:  
        ax.set_title('Impact on lake status (also using P/E)',fontsize=fs)
        ax.annotate('(d) ',(0.05,0.89),fontsize=fs,xycoords='figure fraction',ha='left')

    #Finishing Touches
    #legend = plt.legend(fontsize=fs-1,loc='lower right',markerscale=3,edgecolor='k', handlelength=1,handletextpad=0.3)
    #for handle in legend.legendHandles: handle.set_alpha(1)      # Adjust the transparency of markers
    #legend.get_frame().set_linewidth(0.4)
    ax.set_xlabel('Runoff / '+r'$(E_{{lake}} - P)$'              ,fontsize=fs)
    if q == 'P-E': ax.set_ylabel('(P-E) / '+r'$(E_{{potential}} - P)$'              ,fontsize=fs)
    elif q == 'runoff': ax.set_ylabel('Runoff / '+r'$(E_{{potential}} - P)$'              ,fontsize=fs)
    ax.tick_params(labelsize=fs)    
    #ax.set_title(' ',fontsize=fs)  
    #Save
    plt.tight_layout()
    plt.savefig(wd1+'Figures/Fig4. CCSM4 LGM-PI/LakeLevelScatter_'+age+'_'+qname+'.png',dpi=400)
    plt.show()
    
    
#%%Scatter Plots of lake evap
for age in ['combined']:
    #Scatter plot of evaporation
    dataX = gcm[age]['LM19_lakeEvap']#*(30/10)
    dataY = gcm[age]['PETpr']
    #
    time_bins = range(0,101,10)
    #dataX = dataX.groupby_bins('time', bins=time_bins).mean(dim='time').rename({'time_bins':'time'})
    #dataY = dataY.groupby_bins('time', bins=time_bins).mean(dim='time').rename({'time_bins':'time'})
    #Set up Plot
    plt.figure(figsize=(3,2.5),dpi=400); plt.rc('font', **font)
    ax = plt.axes()
    #Plot
    ax.scatter(dataX,dataY,s=0.5, c= 'teal',lw=0,alpha=0.05,label='All Sites')
    #ax.scatter(dataX.where(np.isfinite(proxyGrid)),dataY.where(np.isfinite(proxyGrid)),s=0.5, c= 'purple',lw=0,alpha=0.3,label='Proxy Sites')
    ax.axline((0,0),(30,30),linestyle='--',c='k',linewidth=0.5,label='1:1 Line')
    ax.set_xlim((0,12)); ax.set_xticks(range(0,13,3)); 
    ax.set_ylim((0,12)); ax.set_yticks(range(0,13,3));
    ax.tick_params(labelsize=6)    
    #Add Skill Annotation
    ax.annotate('r\u00b2 = '+str(da_utils_ls.calcSkill(dataX,dataY,'r2')),#+ ' ('+str(calcSkill(dataX.where(np.isfinite(proxyGrid)),dataY.where(np.isfinite(proxyGrid)),'r2'))+')',
                (0.25,0.9),fontsize=fs,xycoords='axes fraction',ha='center')
    ax.annotate('RMSE = '+str(da_utils_ls.calcSkill(dataX,dataY,'RMSE')),#+ ' ('+str(calcSkill(dataX.where(np.isfinite(proxyGrid)),dataY.where(np.isfinite(proxyGrid)),'RMSE'))+')',
                (0.25,0.83),fontsize=fs,xycoords='axes fraction',ha='center')
    rectangle = patches.Rectangle((0.05, 0.80), 0.4, 0.18, transform=ax.transAxes, facecolor='w', edgecolor='black',lw=0.4)
    #rectangle = patches.Rectangle((0.05, 0.81), 0.5, 0.15, transform=ax.transAxes, facecolor='w', edgecolor='black',lw=0.4)
    ax.add_patch(rectangle)
    #Finishing Touches
    #legend = plt.legend(fontsize=fs,loc='lower right',markerscale=3,edgecolor='k', handlelength=1,handletextpad=0.3)
    #for handle in legend.legendHandles: handle.set_alpha(1)      # Adjust the transparency of markers
    #legend.get_frame().set_linewidth(0.4)
    ax.set_xlabel('Lake (mm/day)',fontsize=fs)
    ax.set_ylabel('Potential (mm/day)',fontsize=fs)
    ax.set_title('Evaporation calculation comparison',fontsize=fs) 
    ax.annotate('(a) ',(0.05,0.89),fontsize=fs,xycoords='figure fraction',ha='left')
    #Save
    plt.tight_layout()
    plt.savefig(wd1+'Figures/Fig4. CCSM4 LGM-PI/EvaporationScatter_'+age+'.png',dpi=400)
    plt.show()
#%%Map Plots

#%%Plot anomalies
age='combined'
fs = 8
geolims = [-120,-35,-50,40]
for i,var in enumerate(['LakeStatusLEVAP','LakeStatusPET','LM19_lakeEvap','PETpr']):
    if i >=2: continue
    name =[['(a) ',r'$\frac{Runoff}{E_{{lake}} - P}$'+' (original)','LowryMorrill'],# '\n(Lowry & Morrill, 2019)',
            ['(b) ',r'$\frac{Runoff}{E_{{potential}} - P}$'+' (for TraCE)',qname], #'\n(Hancock et al., (this Study))',
            ['(a)','Lake Evaporation','Elake'],#\n(Lowry & Morrill, 2019)',
            ['(b)','Potential Evaporation','Epotential']#\n(Hancock et al., (this study))'
            ][i]
    pltvals = gcm[age][var][100:].mean(dim='time') - gcm[age][var][:100].mean(dim='time')
    pltvals = pltvals.where(np.isfinite(gcm[age]['LakeStatusLEVAP'][0]))
    #Set up plot
    #if i == 0: pltvals*=(3*12)
    #else: 
    plt.figure(figsize=(1.5,2.15),dpi=600); plt.rc('font', **font)
    ax = plt.axes(projection=ccrs.Robinson())
    #ax.set_title(name[0],fontsize=fs,loc='left')
    ax.set_title(name[1],fontsize=fs,loc='center')
    da_plot.plotBaseMap(ax,ccrs.Robinson(),lims=geolims)
    ax.add_feature(cfeature.BORDERS,lw=0.4)
    #Plot
    if i > 1: p=ax.pcolormesh(pltvals.lon,pltvals.lat,pltvals,vmin=-2,vmax=2,cmap=cm.get_cmap('BrBG_r',fs),transform=ccrs.PlateCarree())
    else: p = ax.pcolormesh(pltvals.lon,pltvals.lat,pltvals*100,vmin=-80,vmax=80,cmap=cm.get_cmap('BrBG',fs),transform=ccrs.PlateCarree())
    ##r\u00b2 
    ax.annotate(name[0],(0.73,0.8),fontsize=fs,xycoords='axes fraction')

    #Colorbar
    #cbar = plt.colorbar(p,ax=ax,extend='both',shrink=0.9)
    #cbar.ax.tick_params(labelsize=fs-1)
    #if i > 1: cbar.set_label('mm/day (LGM - preIndustrial)',fontsize=fs)
    #else: cbar.set_label('percentile (LGM - preIndustrial)',fontsize=fs)
    #
    #Save
    plt.tight_layout()
    plt.savefig(wd1+'Figures/Fig4. CCSM4 LGM-PI/LakeLevel_Difference_Fig_'+age+'_'+name[2]+'.png',dpi=400)
    plt.show()


#%% 
#Map skill metric 
age='combined'
for method in ['r2']:
    if method == 'r': vmin,vmax,e,name,cmp=0,1,'min','Correlation Coefficient','YlOrRd_r'
    elif method == 'r2': vmin,vmax,e,name,cmp=0,1,None,'r\u00b2','YlOrRd_r'
    elif method=='RMSE':vmin,vmax,e,name,cmp=0,25,'max','RMSE (percentile)','YlOrRd'
    #
    dataX = gcm[age]['LakeStatusLEVAP']*100
    dataY = gcm[age]['LakeStatusPET']*100
    #dataX = gcm[age]['LM19_lakeEvap']#*100
    #dataY = gcm[age]['PETpr']#*100
    #
    #time_bins = range(0,101,10)
    #dataX = dataX.groupby_bins('time', bins=time_bins).mean(dim='time').rename({'time_bins':'time'})
    #dataY = dataY.groupby_bins('time', bins=time_bins).mean(dim='time').rename({'time_bins':'time'})
    skillArray = da_utils_ls.calcSkill(dataX,dataY,method,calcMean=False)
    #
    plt.figure(figsize=(1.5,2.15),dpi=400); plt.rc('font', **font)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_title(name+' with original',fontsize=fs)
    da_plot.plotBaseMap(ax,ccrs.Robinson(),lims=geolims)
    ax.add_feature(cfeature.BORDERS,lw=0.4)
    #Plot
    p = ax.pcolormesh(skillArray.lon,skillArray.lat,skillArray, vmin=vmin,vmax=vmax,cmap=cm.get_cmap(cmp,10),transform=ccrs.PlateCarree())
    ax.scatter(proxy_data['lons'],proxy_data['lats'],transform=ccrs.PlateCarree(),c='w',ec='k',s=2,linewidth=0.3,label='Proxy sites')
    #
    rectangle = patches.Rectangle((0.03, 0.1), 0.5, 0.19, transform=ax.transAxes, facecolor='w', edgecolor='black',lw=0.4,zorder=3)
    ax.add_patch(rectangle)
    ax.annotate('mean '+name+' =\n'+str(da_utils_ls.calcSkill(dataX,dataY,'r2')),#+ ' ('+str(calcSkill(dataX.where(np.isfinite(proxyGrid)),dataY.where(np.isfinite(proxyGrid)),'r2'))+')',
                (0.28,0.13),fontsize=fs,xycoords='axes fraction',ha='center',zorder=3)
    ax.annotate('(c)',(0.73,0.8),fontsize=fs,xycoords='axes fraction')
    #Save
    plt.tight_layout()
    plt.savefig(wd1+'Figures/Fig4. CCSM4 LGM-PI/LakeLevel_r2map_'+age+'_'+qname+'.png',dpi=400)
    plt.show()
    #plt.savefig(wd+'
    #%%