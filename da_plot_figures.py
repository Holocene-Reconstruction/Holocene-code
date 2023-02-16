#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:44:19 2023

@author: chrishancock
"""


#%% VISUALIZE THE OUTPUT - Load Data to plot
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot   as plt         # Packages for making figures
import matplotlib.cm       as cm
import matplotlib.gridspec as gridspec
import cartopy.crs         as ccrs        # Packages for mapping in python
import cartopy.feature     as cfeature
import scipy.stats
import cartopy.util        as cutil
import regionmask as rm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import da_psms

#output_filename = 'holocene_recon_2023-01-26_16_pseudoLakesStationaryPrior'
#output_filename = 'holocene_recon_2023-02-02_10_pseudoLakesTimeVarPrior'
output_dir = '/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP12k/Data/results/'
try:
    os.mkdir(output_dir+output_filename+'/figs/timeseries')
    os.mkdir(output_dir+output_filename+'/figs/anomGrid')
    os.mkdir(output_dir+output_filename+'/figs/skill')
    os.mkdir(output_dir+output_filename+'/figs/summary')
    os.mkdir(output_dir+output_filename+'/figs/proxy')
except: print("folders already created")


save=True
PSMkeys={}
PSMkeys['lakestatusPSM']={
    'c':'slateblue','m':'o', 'units':'percentile','cmap':'BrBG',
    'div':{'name':'archivetype',
           'names':['LakeDeposits','Shoreline'],
           'c':    ['skyblue','slateblue'],
           'm':    ['s','o']
           }
    }
PSMkeys['get_precip']={
    'c':'seagreen','m':'o','units':'mm/a','cmap':'BrBG',
    'div':{'name':'',
           'names':[],
           'c':    [],
           'm':    []
           }
    }
PSMkeys['get_tas']={
    'c':'lightcoral','m':'P','units':'°C','cramp':'seismic',
    }



def proxies2grid(proxyvals,proxylats,proxylons,latlonbins=[5,5],percentile=False):
    lonvec,latvec = range(0,361,latlonbins[1]),range(-90,91,latlonbins[0])
    valsXY = xr.DataArray(data=np.zeros((len(proxyvals.ages),len(proxyvals.proxy),len(latvec),len(lonvec)))*np.NaN, dims=["ages","proxy","lat", "lon"],coords={'ages':proxyvals.ages,'proxy':proxyvals.proxy,'lat':latvec,'lon':lonvec})
    # Bin by lat/lonvec
    for i in range(len(proxyvals.proxy)):
        proxyLats_i = np.argmin(abs(latvec-proxylats[i]))
        proxyLons_i = np.argmin(abs(lonvec-proxylons[i]))
        if percentile == 'lakestatusPSM': valsXY[:,i,proxyLats_i,proxyLons_i] = da_psms.vals2percentile(proxyvals[:,i])
        else:  valsXY[:,i,proxyLats_i,proxyLons_i] = proxyvals[:,i]
    return(valsXY)
   
def plotTSxDim(data,labels,title,dim,keys,seasons,
                 save=False,save_dir=output_dir+output_filename+'/figs/'):
    #
    ncol=len(seasons); nrow=int(len(data))
    if nrow == 1: plt.figure(figsize=(ncol*6,nrow*5))
    else: plt.figure(figsize=(ncol*6,nrow*2))
    axes = np.empty(shape=(nrow, ncol), dtype=object)
    gs = gridspec.GridSpec(nrow,ncol)
    for c, szn in enumerate(seasons):
        for r,vals in enumerate(data):
            ax = plt.subplot(gs[nrow-(r+1),c],sharey=axes[r, 0]); axes[r, c] = ax
            for j,key in enumerate(keys): 
                if key[:5]   == 'recon': col  ='firebrick'
                elif key[:5] == 'prior': col  ='grey'
                elif key[:5] == 'input': col  ='royalblue'
                elif key[:5] == 'proxy': col  ='goldenrod'
                else: col='deeppink'
                if key[-3:].lower() != 'ens': ax.plot(vals[szn][key].ages,vals[szn][key],c=col,label=key+'_'+szn)
                else: ax.fill_between(vals[szn][key].ages, vals[szn][key].max(dim='ens_selected'), vals[szn][key].min(dim='ens_selected'), facecolor=col, alpha=0.25)
                #
                ax.invert_xaxis(); ax.set_xlim(max(vals[szn][key].ages), min(vals[szn][key].ages))
            if c == 0: 
                ax.set_ylabel(labels[r])
            if r == 0: ax.set_xlabel("Age"); ax.legend(ncol=3,bbox_to_anchor=(0.5, -0.5,0,0),loc='center');  ax.xaxis.set_ticks([*range(0,22000,3000)])
            else: ax.xaxis.set_ticklabels([])
    #
    plt.suptitle(title)
    plt.tight_layout()
    if save == True: plt.savefig(save_dir+title+'.png',dpi=300,bbox_inches='tight')
    plt.show()
    
def plotAnomGrid(dataList,timevec,labels,cmap,title,vlist,cbarXrow='All',
                 view='contour',scatter=False,proj=ccrs.Robinson(),
                 save=False,save_dir=output_dir+output_filename+'/figs/'):
    #
    s=2
    ncol=3; nrow=int(np.ceil(len(dataList)/ncol))
    plt.figure(figsize=(ncol*3,nrow*2),dpi=400)
    gs = gridspec.GridSpec(nrow+1,ncol*s+1)
    #
    #
    for t,vals in enumerate(dataList):
        if len(vlist) > 1: v= vlist[t]
        else: v = vlist[0]
        cmap2=cmap
        if v[0] > v[1]: 
            v=[v[1],v[0]]; 
            if cmap[-2:]=='_r': cmap2=cmap[:-2]
            else: cmap2 = cmap+'_r'
        vlev = np.round(np.linspace(v[0],v[1],11),2)
        r = int(np.floor(t/ncol)); c = int((t%ncol)*s)
        #
        if proj == False:  ax = plt.subplot(gs[r,c:c+s]) 
        else:              ax = plt.subplot(gs[r,c:c+s],projection=proj);    ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.add_feature(cfeature.COASTLINE, linewidth=0.3);ax.add_feature(cfeature.LAND,color='whitesmoke')
        #
        if view == 'grid': 
            p=ax.pcolormesh(vals.lon,vals.lat,vals,vmin=v[0],vmax=v[1],cmap=cmap2,transform=ccrs.PlateCarree())
        if view == 'gridProxy': 
            p=ax.pcolormesh(vals.lon,vals.lat,vals,vmin=v[0],vmax=v[1],cmap=cmap2,alpha=1,transform=ccrs.PlateCarree())
        elif view == 'contour': 
            data_cyclic,lon_cyclic = cutil.add_cyclic_point(vals,coord=vals.lon.data)
            p=ax.contourf(lon_cyclic,vals.lat,data_cyclic, cmap=cm.get_cmap(cmap2, 11-1),levels=vlev,extend='both',transform=ccrs.PlateCarree())
        elif view == 'scatter':
            vals2 = np.concatenate(scatter[t],axis=0)
            lats  = np.repeat(scatter[t].lat,len(scatter[t].lon)).data[~np.isnan(vals2)]
            lons  = np.array(list(scatter[t].lon.data)*len(scatter[t].lat))[~np.isnan(vals2)]
            vals2 = vals2[~np.isnan(vals2)]
            p=ax.scatter(lons,lats,c=vals2,vmin=v[0],vmax=v[1],marker='.',ec='k',cmap=cmap2,transform=ccrs.PlateCarree())
            #ax.
        ax.set_title(labels[t],fontsize='xx-small')
        if cbarXrow == 'row':  #ColorBar
            if c == ncol-1:
                ax = plt.subplot(gs[r,ncol*2])
                ax.set_axis_off()
                if v==[-1, 1]: e = 'neither'
                elif v==[0,1]: e = 'min'
                elif v[0] == 0: e='max' 
                else: e='both' 
                cbar = plt.colorbar(p,orientation='vertical',extend=e,cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
                cbar.set_label(title[3]+title[4],fontsize='xx-small',c='black')
                cbar.ax.tick_params(labelsize='xx-small') 
    if cbarXrow == 'All':  #ColorBar
        ax = plt.subplot(gs[:nrow,ncol*2])
        ax.set_axis_off()
        cbar = plt.colorbar(p,orientation='vertical', extend='both',cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
        cbar.set_label(title[3]+title[4],fontsize='xx-small',c='black')
        cbar.ax.tick_params(labelsize='xx-small') 
    plt.suptitle(title[2]+' '+title[1].upper()+' '+title[0].upper(),fontsize='x-small')
    #
    plt.tight_layout()
    if save == True: plt.savefig(save_dir+title[2]+'_'+title[0]+'_'+title[1]+'.png',dpi=600,bbox_inches='tight')
    plt.show()
   
handle = xr.open_dataset(output_dir+output_filename+'/'+output_filename+'.nc',decode_times=False)
proxyData={}
for key in ['proxies_assimilated','proxy_metadata','proxy_values','proxy_resolutions']:
    proxyData[key]=handle[key]
    
proxyData['values_binned']     = np.transpose(proxyData['proxy_values'].data)
proxyData['resolution_binned'] = np.transpose(proxyData['proxy_resolutions'].data)

for key in ['lats','lons','seasonality_array','proxySzn','units','archivetype','proxytype','proxyPSM','interp']:proxyData[key]=[]
for proxy in range(len(proxyData['proxy_metadata'])): #refer to load proxies line 390ish
    proxyData['lats'].append(       float(proxyData['proxy_metadata'][proxy][2]))
    proxyData['lons'].append(       float(proxyData['proxy_metadata'][proxy][3]))
    proxyData['seasonality_array'].append([int(x) for x in str(proxyData['proxy_metadata'][proxy][4].data).replace('[','').replace(']','').replace(' ',',').replace(',,',',').split(',') if x != ''])
    proxyData['proxySzn'].append  (   str(proxyData['proxy_metadata'][proxy][5].data))
    proxyData['units'].append(        str(proxyData['proxy_metadata'][proxy][8].data))
    proxyData['archivetype'].append(  str(proxyData['proxy_metadata'][proxy][9].data))
    proxyData['proxytype'].append(    str(proxyData['proxy_metadata'][proxy][10].data))
    proxyData['proxyPSM'].append(     str(proxyData['proxy_metadata'][proxy][11].data))
    proxyData['interp'].append(  str(proxyData['proxy_metadata'][proxy][12].data))
for key in ['lats','lons','proxySzn','units','archivetype','proxytype','proxyPSM','interp']:proxyData[key]=np.array(proxyData[key])

handle.close()

#%%
handle = xr.open_dataset(output_dir+output_filename+'/'+output_filename+'.nc',decode_times=False)

keys = ['recon','prior','input','reconEns','priorEns']

dampVars = {
    'tas':{       'data':[],'cmap':'seismic','proxy':False},
    'precip':{    'data':[],'cmap':'BrBG',   'proxy':False},
    'LakeStatus':{'data':[],'cmap':'BrBG',   'proxy':'lakestatusPSM'}
    #'U250':{      'data':[],'cmap':'PuOr_r', 'proxy':False},
    #'ITCZprecip':{'data':[],'cmap':'BrBG',   'proxy':False}
    }

land=False
if land:
    landFrac = xr.open_dataset("/Users/chrishancock/Library/CloudStorage/OneDrive-NorthernArizonaUniversity/Research/Manuscript/DAMP12k/Data/models/DAMP12kTraCE/trace.01-36.22000BP.cam2.LANDFRAC.22000BP_decavg_400BCE.nc")
    landFrac = landFrac['LANDFRAC'].sum(dim='time')/len(landFrac['LANDFRAC'].time)

#
for var in dampVars.keys():
    print(var)
    if var == 'LakeStatus': seasons=['annual']
    else: seasons=['annual','jja','djf']
    dampVars[var]['units'] = ' ('+handle['units_'+var+'_annual'].data[0]+')'
    dampVars[var]['seasons'] = seasons
    damp={}
    for key in keys:
        damp[key] = []
        if key[-3:].lower() == 'ens': 
            for szn in seasons: damp[key].append(handle[key[:-3]+'_'+var+'_'+szn+'_ens'])
            damp[key] = xr.concat(damp[key],dim='season').assign_coords(season=seasons).transpose("ages", "season","ens_selected","lat","lon")
        else:                         
            for szn in seasons: damp[key].append(handle[key     +'_'+var+'_'+szn+'_mean'])
            damp[key] = xr.concat(damp[key],dim='season').assign_coords(season=seasons).transpose("ages", "season","lat","lon")            
        if dampVars[var]['units'] == ' (latitude)': 
            if 'ens_selected' in damp[key].dims: damp[key]=damp[key][:,:,:,0,:]
            else: damp[key]=damp[key][:,:,0,:]
        elif  dampVars[var]['units'] == ' (index)':  
            if 'ens_selected' in damp[key].dims: damp[key]=damp[key][:,:,:,0,0]
            else: damp[key]=damp[key][:,:,0,0]
        if land:  
            damp[key]= damp[key].where(landFrac.data>0.5)
       # if var == 'LakeStatus': 
            #damp[key]=(damp[key]-damp[key].mean(dim='ages'))/damp[key].std(dim='ages')
            #for lat in range(len(damp[key].lat)): 
             #   for lon in range(len(damp[key].lon)):
              #      if 'ens_selected' in damp[key].dims: 
               #         for i in range(len(damp[key].ens_selected)):
                #            damp[key][:,0,i,lat,lon] = vals2percentile(damp[key][:,0,i,lat,lon])  
                 #   else:   damp[key][:,0,  lat,lon] = vals2percentile(damp[key][:,0,  lat,lon])  
    dampVars[var]['data']  = damp 

handle.close()
if land: landFrac.close()
        #



#%%
plotAnoms=False ; plotSkill=False ; plotTS=False ; summaryPlot=True
save=save

for var in  ['tas','LakeStatus']:
    damp,units,seasons,cmap,proxy = dampVars[var]['data'],dampVars[var]['units'],dampVars[var]['seasons'], dampVars[var]['cmap'],dampVars[var]['proxy']
    #Plot Maps
    if plotAnoms:
        #if units == ' (latitude)': continue
        bs=2000
        labels  = []
        timevec = [[-x+bs/2,-x-bs/2] for x in range(-18000,0,bs)]; 
        basevec = [bs,0-bs/2]
        for szn in seasons:
            for key in ['recon','input','update']:
                if key == 'update':vals= damp['recon'].sel(season=szn) - damp['input'].sel(season=szn)
                else: vals = damp[key].sel(season=szn)*1
                if len(np.shape(vals)) == 3:
                    dataList, labels = [], []
                    title = [var,szn,key,'anomaly',units]
                    if land: title[0] += '_landOnly'
                    #Get Plot values
                    basevals=  vals.sel(ages=slice(basevec   [1],basevec   [0])).mean(dim='ages') 
                    for i,times in enumerate(timevec):
                        dataList.append(vals.sel(ages=slice(times[1],times[0])).mean(dim='ages') - basevals)
                        labels.append(str(int(np.mean(times)/1000))+'-'+str(int(np.mean(basevec)/1000))+' ka [+/-'+str(bs/2000)+']')
                    #Replace last map
                    dataList[-1] = basevals-vals.mean(dim='ages')
                    labels[-1] = str(int(np.mean(basevec)/1000))+' ka'+'[+/-'+str(bs/2000)+']'+' - mean'
                    #Plot
                    v = np.nanquantile(np.abs([x.data for x in dataList]),0.9)
                    plotAnomGrid(dataList,timevec,labels,cmap,title,vlist=[[-v,v]],view='grid',cbarXrow='All',
                                 save=save,save_dir=output_dir+output_filename+'/figs/anomGrid/')
    #
    #Plot "skill" metrics comparing input and reconstruction #"skill" is only a skill metric for pseudo proxies.  #otherwise it's a comparison  of the reconstruction and trace timeseries agreement
    if plotSkill:     
        for szn in seasons:
            ages   = damp['recon'].ages.data
            timevec=[[np.argmin(ages),np.argmax(ages)],[np.argmin(abs(ages-np.median(ages))),np.argmax(ages)],[np.argmin(ages),np.argmin(abs(ages-np.median(ages)))]]
            #Calc skill metrics
            dataList,v,labels,scatter = [],[],[],[]
            title = [var,szn,'input vs recon','skill metrics',units]
            for i,times in enumerate(timevec):
                t1,t2 = ages[times].data
                vals1,vals2 = damp['input'].sel(season=szn).sel(ages=slice(t1,t2+1)), damp['recon'].sel(season=szn).sel(ages=slice(t1,t2+1))
                for method in ['Correlation','CE','RMSE']:
                    if   method == 'Correlation': 
                        vals = xr.corr(vals1,vals2,dim='ages'); 
                        v.append([-1,1])
                    elif method == 'CE': 
                        vals =1-((vals2-vals1)**2).sum(dim='ages',skipna=False)/((vals2-vals1.mean(dim='ages'))**2).sum(dim='ages',skipna=False);
                        v.append([0,1])
                    elif method == 'RMSE': 
                        vals = np.sqrt(((vals2 - vals1) ** 2).mean(dim='ages')); 
                        v.append([np.nanquantile(damp['input'].sel(season=szn).var(dim='ages')**0.5,0.5),0])
                    g = np.round(vals.weighted(np.cos(np.deg2rad(vals.lat))).mean(dim=["lon", "lat"]).data,2)
                    dataList.append(vals)
                    labels.append(str(int(t2/1000))+'-'+str(int(t1/1000))+' ka \n '+method+' ('+str(g)+')')
            #Reorder to how we want to plot
            v=[v[x] for x in [0,3,6,1,4,7,2,5,8]]
            labels =[labels[x] for x in [0,3,6,1,4,7,2,5,8]]
            dataList=[dataList[x] for x in [0,3,6,1,4,7,2,5,8]]
            #Plot
            plotAnomGrid(dataList,timevec,labels,'Spectral_r',title,vlist=v,view='grid',cbarXrow='row',
                                         save=save,save_dir=output_dir+output_filename+'/figs/skill/')
        #
    #Plot Timeseries
    keys=['recon','prior','input','reconEns','priorEns']
    if plotTS:
        if units == ' (latitude)':  dim='lon';binvec=np.linspace(0,360,7)
        else: dim='lat';  binvec=np.linspace(-90,90,7)
        title=var+' by '+dim+units.replace('/','.')
        tital_global = 'global ' +  title 
        if land: title+=' landOnly'
        #
        dataList, labels = [{} for x in binvec], []
        if proxy != False: 
            idx  = (proxyData['proxyPSM']==proxy) & (proxyData['proxies_assimilated'].sum(dim='ages') > 0)
            if(sum(idx)>0):keys+=['proxy']
        for i in range(len(binvec)):
            if binvec[i]==binvec[-1]: t1,t2 = binvec[0],binvec[-1]; labels.append('Global')
            else: t1,t2=binvec[i],binvec[i+1]; labels.append(str(int(t1))+' - '+str(int(t2))+'° '+dim)
            #
            for szn in seasons:
                dataList[i][szn]={} 
                for key in keys:
                    if key == 'proxy':
                        vals = proxies2grid(proxyData['proxy_values'][:,idx],proxyData['lats'][idx],proxyData['lons'][idx],latlonbins=[3,30],percentile=False)
                        vals = vals.mean(dim='proxy').sel(lat=slice(t1,t2))
                    else:
                        if dim =='lat':  vals = damp[key].sel(season=szn).sel(lat=slice(t1,t2))
                        else:            vals = damp[key].sel(season=szn).sel(lon=slice(t1,t2))
                    if dim =='lat': vals = vals.weighted(np.cos(np.deg2rad(vals.lat))).mean(dim=["lon", "lat"])
                    else:           vals = vals.mean("lon")
                    #
                    dataList[i][szn][key] = vals-vals[0]
        #
        plotTSxDim(dataList,labels,title,dim,keys,seasons,
                   save=save,save_dir=output_dir+output_filename+'/figs/timeseries/')
        plotTSxDim([dataList[-1]],[labels[-1]],tital_global,dim,keys,seasons=['annual'],
                   save=save,save_dir=output_dir+output_filename+'/figs/timeseries/')
    #
    #Summary Plot 
    if summaryPlot:
        plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(4,2)#,wspace=0.3,hspace=0.4)
        #Plot TS###########################################################################
        ax = plt.subplot(gs[0,0]) 
        ax.title.set_text('Global Annual Mean')
        keys,colors = ['prior','input','recon'],['grey','indigo','teal']
        for i,key in enumerate(keys):
            if units == ' (latitude)': vals = damp[key].sel(season='annual').mean(("lon"))
            else: vals = damp[key].sel(season='annual').weighted(np.cos(np.deg2rad(damp[key].lat))).mean(("lon", "lat"))
            ax.plot(vals.ages,vals,c=colors[i],lw=3,label=key[:5])
            try: 
                if units == ' (latitude)': valsEns = damp[key+'Ens'].sel(season='annual').mean(("lon"))
                else: valsEns = damp[key+'Ens'].sel(season='annual').weighted(np.cos(np.deg2rad(damp[key].lat))).mean(("lon", "lat"))
                ax.fill_between(vals.ages, np.nanmax(valsEns,axis=1), np.nanmin(valsEns,axis=1), facecolor=colors[i], alpha=0.3)
            except: continue
        ax.set_xlim(np.quantile(damp[key]['ages'],[0,1])),ax.set_xlabel("Age (ka)"); ax.invert_xaxis(); 
        ax.xaxis.set_ticks([*range(0,22000,3000)]); ax.set_xticklabels([*range(0,22,3)])
        ax.legend()
        #Plot recon by season###########################################################################
        ax = plt.subplot(gs[0,1]) 
        ax.title.set_text('Reconstruction by Season')
        key,colors = 'recon',['grey','firebrick','royalblue']
        for i,szn in enumerate(seasons):
            if units == ' (latitude)': vals = damp[key].sel(season=szn).mean(("lon"))
            else: vals = damp[key].sel(season=szn).weighted(np.cos(np.deg2rad(damp[key].lat))).mean(("lon", "lat"))
            ax.plot(vals.ages,vals,c=colors[i],lw=3,label=szn)
            try: 
                if units == ' (latitude)': valsEns = damp[key+'Ens'].sel(season=szn).mean(("lon"))
                else: valsEns = damp[key+'Ens'].sel(season=szn).weighted(np.cos(np.deg2rad(damp[key].lat))).mean(("lon", "lat"))
                ax.fill_between(vals.ages, valsEns.max(dim='ens_selected'), valsEns.min(dim='ens_selected'), facecolor=colors[i], alpha=0.3)
            except: continue
        ax.set_xlim(np.quantile(damp[key]['ages'],[0,1])),ax.set_xlabel("Age (ka)"); ax.invert_xaxis(); 
        ax.xaxis.set_ticks([*range(0,22000,3000)]); ax.set_xticklabels([*range(0,22,3)])
        ax.legend()       
        #Plot Anomalies###########################################################################
        key,szn,bs,='recon','annual',1000
        if units == ' (latitude)':  
            dataVals = damp[key]*1
        else:                       
            dataVals = damp[key].sel(season=szn)*1
            v=np.nanquantile(np.abs(dataVals-dataVals.sel(ages=min(damp[key]['ages']))),0.85)
            v=np.round(np.linspace(-v,v,21),2)
        vals0 = dataVals.sel(ages=slice(0.5-bs/2,0.5+bs/2)).mean(dim='ages')
        for i,time in enumerate([20000,6000]):
            ax = plt.subplot(gs[1,i],projection=ccrs.Robinson()) 
            ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(lw=0.2)
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.2, color='k', alpha=0.4, linestyle=(0,(5,10)))
            if units == ' (latitude)':
                vals = dataVals.sel(ages=slice(time-bs/2,time+bs/2)).mean(dim='ages')
                ax.fill_between(vals0.lon, vals0.sel(season='jja'), vals0.sel(season='djf'), facecolor='firebrick', alpha=0.5,transform=ccrs.PlateCarree())
                ax.fill_between(vals.lon, vals.sel(season='jja'), vals.sel(season='djf'), facecolor='royalblue', alpha=0.5,transform=ccrs.PlateCarree())
                ax.plot(vals0.lon,vals0.sel(season='annual'),color='firebrick',linestyle=':',transform=ccrs.PlateCarree(),label='0ka')
                ax.plot(vals.lon,vals.sel(season='annual'),color='royalblue',linestyle='-.',transform=ccrs.PlateCarree(),label=str(time)+'ka')
                ax.legend()
            else:
                vals = dataVals.sel(ages=slice(time-bs/2,time+bs/2)).mean(dim='ages')-vals0 
                data_cyclic,lon_cyclic = cutil.add_cyclic_point(vals,coord=vals.lon.data)
                p=ax.contourf(lon_cyclic,vals.lat,data_cyclic, cmap=cm.get_cmap(cmap, 21-1),levels=v,extend='both',transform=ccrs.PlateCarree())
                cbar = plt.colorbar(p,orientation='vertical', extend='both')#,cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
                cbar.set_label('Annual Anomaly'+units,fontsize='small',c='black')
                cbar.ax.tick_params(labelsize='x-small') 
            ax.title.set_text(str(int(time/1000))+'-0 ka (+/-'+str((bs/2)/1000)+' ka)')
        #Plot Proxies Used###########################################################################
        ax = plt.subplot(gs[2,0],projection=ccrs.Robinson()) 
        ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(linewidth=0.3)
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.2, color='k', alpha=0.4, linestyle=(0,(5,10)))
        ax.set_title('Proxy Data (grouped by PSM)')
        #
        ax1 = plt.subplot(gs[2,1])
        ax1.title.set_text('Proxy density over time')
        ax1.set_xlim(np.quantile(damp[key]['ages'],[0,1])),ax1.set_xlabel("Age (ka)"); ax1.invert_xaxis(); ax1.xaxis.set_ticks([*range(0,22000,3000)]); ax1.set_xticklabels([*range(0,22,3)])
        ax1.set_ylabel('Count');
        #
        ages=proxyData['proxies_assimilated'].ages
        vals=ages*0
        for i,PSM in enumerate(np.unique(proxyData['proxyPSM'])):
            idx=(proxyData['proxyPSM']==PSM) & (proxyData['proxies_assimilated'].sum(axis=0)>0)
            ax.scatter(proxyData['lons'][idx],proxyData['lats'][idx],s=20,alpha=0.4,c=PSMkeys[PSM]['c'],marker=PSMkeys[PSM]['m'],ec='k',label=PSM+' ('+str(int(sum(idx)))+')',transform=ccrs.PlateCarree())
            valsNew = proxyData['proxies_assimilated'][:,proxyData['proxyPSM']=='lakestatusPSM'].sum(dim='proxy')
            if i == 0: ax1.bar(ages, valsNew,              color=PSMkeys[PSM]['c'], width=np.nanmedian(np.diff(ages)),label=PSM)
            else:      ax1.bar(ages, valsNew, bottom=vals, color=PSMkeys[PSM]['c'], width=np.nanmedian(np.diff(ages)),label=PSM)
            vals+=valsNew
        ax1.legend(loc='lower right')
        ############################################################################
        #############################################################################
        method,szn = 'Correlation','annual'
        keys = ['input','recon']
        vals1,vals2 = damp[keys[0]].sel(season=szn), damp[keys[1]].sel(season=szn)        
        if proxy:
            idx=((proxyData['proxyPSM']==proxy) & (proxyData['proxies_assimilated'].sum(dim='ages')==0))
            if sum(np.isfinite(idx)) == 0: proxy = False
            else: 
                dataVals = {'proxy':proxyData['proxy_values'][:,idx]}
                for key in keys:
                    model_vals = {var:damp[key].values,'lat':damp[key].lat.values,'lon':damp[key].lon.values,'season':seasons}
                    dataVals[key] = dataVals['proxy']*np.NaN; 
                    count=0
                    pseudoLakes= da_psms.psm_main(model_vals,proxyData,[]);
                    for i,tf in enumerate(idx):
                        if tf: 
                            dataVals[key][:,count]=pseudoLakes[0][i][list(pseudoLakes[0][i].keys())[-1]];
                            dataVals[key][np.isnan(dataVals['proxy'][:,count]),count] = np.NaN
                            dataVals[key][:,count] = da_psms.vals2percentile(dataVals[key][:,count])
                            dataVals[key][:,count] -=  np.nanmean(dataVals[key][:3,count])
                            count+=1
        #        
        ax = plt.subplot(gs[3,0],projection=ccrs.Robinson()) 
        ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(lw=0.3)
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.2, color='k', alpha=0.4, linestyle=(0,(5,10)))
        #
        if method == 'Correlation':
            vals = xr.corr(vals1,vals2,dim='ages'); 
            v=1; newunits = 'Correlation Coefficient'; e='neither'
            if units == ' (latitude)':
                vals=vals.expand_dims({'lat':[0]})
                p=ax.pcolormesh(vals.lon,[-4,0,4],np.repeat(vals.data,3,axis=0), cmap='Spectral_r',vmin=-v,vmax=v,transform=ccrs.PlateCarree())
            else: 
                p=ax.pcolormesh(vals.lon,vals.lat,vals, cmap='Spectral_r',vmin=-v,vmax=v,transform=ccrs.PlateCarree())
            if proxy:
                valsP1 = xr.corr(dataVals['proxy'],dataVals[keys[0]],dim='ages')
                valsP2 = xr.corr(dataVals['proxy'],dataVals[keys[1]],dim='ages')
                ax.scatter(proxyData['lons'][idx][np.isfinite(valsP2)],proxyData['lats'][idx][np.isfinite(valsP2)],c=valsP2[np.isfinite(valsP2)],ec='k',cmap='Spectral_r',vmin=-v,vmax=v,transform=ccrs.PlateCarree())
        elif method == 'CE': 
            vals = 1-((vals2-vals1)**2).sum(dim='ages',skipna=False)/((vals2-vals1.mean(dim='ages'))**2).sum(dim='ages',skipna=False);
            v = 1; newunits = 'CE'+units; e='min'
            p=ax.pcolormesh(vals.lon,vals.lat,vals, cmap='Spectral_r',vmin=0,vmax=v,transform=ccrs.PlateCarree())
            if proxy:
                valsP1 = 1-((dataVals['proxy']-dataVals[keys[0]])**2).sum(dim='ages',skipna=False)/((dataVals['proxy']-dataVals[keys[0]].mean(dim='ages'))**2).sum(dim='ages',skipna=False);
                valsP2 = 1-((dataVals['proxy']-dataVals[keys[1]])**2).sum(dim='ages',skipna=False)/((dataVals['proxy']-dataVals[keys[1]].mean(dim='ages'))**2).sum(dim='ages',skipna=False);
                ax.scatter(proxyData['lons'][idx][np.isfinite(valsP2)],proxyData['lats'][idx][np.isfinite(valsP2)],c=valsP2[np.isfinite(valsP2)],ec='k',cmap='Spectral_r',vmin=0,vmax=v,transform=ccrs.PlateCarree())
        elif method == 'RMSE':
            vals = np.sqrt(((vals2 - vals1) ** 2).mean(dim='ages'))
            v = np.nanquantile(damp['recon'].sel(season=szn).var(dim='ages')**0.5,0.95); newunits = 'RMSE'+units; e='max'
            p=ax.pcolormesh(vals.lon,vals.lat,vals, cmap='Spectral',vmin=0,vmax=v,transform=ccrs.PlateCarree())
            if proxy:
                valsP1 = np.sqrt(((dataVals['proxy'] - dataVals[keys[0]]) ** 2).mean(dim='ages'))
                valsP2 = np.sqrt(((dataVals['proxy'] - dataVals[keys[1]]) ** 2).mean(dim='ages'))
                ax.scatter(proxyData['lons'][idx][np.isfinite(valsP2)],proxyData['lats'][idx][np.isfinite(valsP2)],c=valsP2[np.isfinite(valsP2)],ec='k',cmap='Spectral',vmin=0,vmax=v,transform=ccrs.PlateCarree())
        #print(np.round(vals.weighted(np.cos(np.deg2rad(vals.lat))).mean(dim=["lon", "lat"]).data,2))
        ax.title.set_text('Input v. Recon (avg='+str(np.round(np.nanmedian(vals),2))+')')
        cbar = plt.colorbar(p,orientation='vertical',extend=e)#,cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
        cbar.ax.tick_params(labelsize='x-small') 
        cbar.set_label(newunits,fontsize='small',c='black')
        if proxy:
            ax = plt.subplot(gs[3,1])
            counts, bins = np.histogram(valsP1[np.isfinite(valsP1)],range=(np.nanmin([valsP2,valsP1]),np.nanmax([valsP2,valsP1])))
            ax.stairs(counts, bins,color='indigo',fill=True,label=keys[0]+' (avg = '+str(float(np.round(np.mean(valsP1),2)))+')',alpha=0.4)
            counts, bins = np.histogram(valsP2[np.isfinite(valsP2)],range=(np.nanmin([valsP2,valsP1]),np.nanmax([valsP2,valsP1])))
            ax.stairs(counts, bins,color='teal',fill=True,label=keys[1]+' (avg = '+str(float(np.round(np.mean(valsP2),2)))+')',alpha=0.4)
            ax.legend()
            ax.set_xlabel(newunits)
            ax.set_ylabel('Count'); 
            ax.title.set_text(method+' of proxies withheld')
        ############################################################################
        plt.suptitle(var.upper()+units)
        plt.tight_layout()
        if save:  plt.savefig(output_dir+output_filename+'/figs/summary/'+var+'_summary.png',dpi=600,bbox_inches='tight')
        plt.show()
        #


   #%%



      
bs,latlonbins = 2000,[5,5]
timevec       = [[-x+bs/2,-x-bs/2] for x in range(-18000,0,bs)]; 
basevec = [bs,0-bs/2]

plot_proxiesGrid=True
if plot_proxiesGrid: 
    for PSM in np.unique(proxyData['proxyPSM']):
        dataList, labels = [], []
        newunits='percentile'
        title = [PSM,'allseasons','proxy_'+str(latlonbins[0])+'x'+str(latlonbins[1])+'°','anomaly',' ('+PSMkeys[PSM]['units']+')']
        #Load Proxy Data
        idx = (proxyData['proxyPSM']=='lakestatusPSM') & (proxyData['proxies_assimilated'].sum(dim='ages') > 0)
        if PSM == 'lakestatusPSM': vals= proxies2grid(proxyData['proxy_values'][:,idx],proxyData['lats'][idx],proxyData['lons'][idx],latlonbins=latlonbins,percentile=False)
        else:                      vals= proxies2grid(proxyData['proxy_values'][:,idx],proxyData['lats'][idx],proxyData['lons'][idx],latlonbins=latlonbins,percentile=False)
        #
        basevals=vals.sel(ages=slice(basevec[1],basevec[0])).mean(dim=['ages','proxy'])
        vals=vals.mean(dim='proxy')
        for i,times in enumerate(timevec):
            dataList.append(vals.sel(ages=slice(times[1],times[0])).mean(dim='ages') - basevals)
            labels.append(str(int(np.mean(times)/1000))+'-'+str(int(np.mean(basevec)/1000))+' ka [+/-'+str(bs/2000)+']')
        dataList[-1] = basevals-vals.mean(dim='ages')
        labels[-1]   = str(int(np.mean(basevec)/1000))+' ka'+'[+/-'+str(bs/2000)+']'+' - mean'
        v = np.nanquantile(np.abs([x.data for x in dataList]),0.85)
        plotAnomGrid(dataList,timevec,labels,cmap=PSMkeys[PSM]['cmap'],title=title,vlist=[[-v,v]],view='gridProxy',scatter=False,cbarXrow='All',
                     save=save,save_dir=output_dir+output_filename+'/figs/anomGrid/')

plot_proxiesUsed=True
if plot_proxiesUsed: 
    plt.figure(figsize=(10,10),dpi=600)
    gs = gridspec.GridSpec(3,len(np.unique(proxyData['proxyPSM']))) 
    for PSM in np.unique(proxyData['proxyPSM']):
        idx=np.where((proxyData['proxyPSM']==PSM))[0]# & (proxyData['proxies_assimilated'].sum(axis=0)!=0))[0]
        #Map by archive or proxy type
        ax = plt.subplot(gs[0,0],projection=ccrs.Robinson()) 
        ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines()
        for i in range(len(PSMkeys[PSM]['div']['names'])):
            div= PSMkeys[PSM]['div']['names'][i]
            idx2 = [x for x in np.where(proxyData[PSMkeys[PSM]['div']['name']]==div)[0] if x in idx] 
            ax.scatter(proxyData['lons'][idx2],proxyData['lats'][idx2],c=PSMkeys[PSM]['div']['c'][i],ec='k',alpha=0.7,marker=PSMkeys[PSM]['div']['m'][i],label=div,transform=ccrs.PlateCarree())
        ax.legend(bbox_to_anchor=(1, 0.5, 0, 0))
        #Map by season
        idx2 ={
            'Annual':{'idx':np.array([x for x in np.where(np.char.find(proxyData['proxySzn'], 'Ann')==0)[0] if x in idx]),'c':'k','m':'o'},
            'Summer':{'idx':np.array([x for x in np.where(np.char.find(proxyData['proxySzn'], 'sum')==0)[0] if x in idx]),'c':'red','m':'^'},
            'Winter':{'idx':np.array([x for x in np.where(np.char.find(proxyData['proxySzn'], 'win')==0)[0] if x in idx]),'c':'blue','m':'v'}
            }
        ax = plt.subplot(gs[1,0],projection=ccrs.Robinson()) 
        ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines()
        for szn in idx2.keys():
           if len(idx2[szn]['idx']) > 0 :
                ax.scatter(proxyData['lons'][idx2[szn]['idx']],proxyData['lats'][idx2[szn]['idx']],c=idx2[szn]['c'],marker=idx2[szn]['m'],label=szn,transform=ccrs.PlateCarree())
        ax.legend(bbox_to_anchor=(1, 0.5, 0, 0))
        #Map by record length:
        ax = plt.subplot(gs[2,0],projection=ccrs.Robinson()) 
        ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines()
        p=ax.scatter(proxyData['lons'],proxyData['lats'],cmap='magma',ec='k', c=(proxyData['proxies_assimilated'].sum(dim='ages')/len(proxyData['proxies_assimilated'].ages))*100,transform=ccrs.PlateCarree())
        plt.colorbar(p,orientation='vertical', cax=inset_axes(ax,width="5%",height="90%",bbox_to_anchor=(0.1,0,1,1),bbox_transform=ax.transAxes, loc="center right")).set_label('Record Length (% of reconstruction)',c='black')
        #Finalzie figure
        plt.suptitle(PSM)
        plt.tight_layout()
        if save: plt.savefig(output_dir+output_filename+'/figs/proxies_'+PSM+'.png',dpi=600,bbox_inches='tight')
        plt.show()
    
        
