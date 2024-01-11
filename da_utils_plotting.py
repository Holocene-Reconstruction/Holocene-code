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

def loadDAMPresults(path,filename):
    #Load Options
    options = {}
    with open(path+filename+'/options.txt','r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.strip().split(',')
            key, value = l[0], l[1:]
            if len(l[1:]) > 1: value = [x.replace("[","").replace("]","").replace("'","").replace('"',"").replace(' ',"") for x in value]
            try: value = [float(x) for x in value]
            except: value = value
            if len(value) == 1: value = value[0]
            if value == 'True': value = True
            if value == 'False': value = False   
            options[key] = value
    #
    #Load DA Reconstruction
    handle = xr.open_dataset(path+filename+'/'+filename+'.nc',decode_times=False)
    #Gridded Data
    DAMPvals={}
    print('DA reconstruction loaded for variables:')
    for var_name in options['vars_to_reconstruct']: 
        print(var_name+' ['+handle['units_'+var_name].data+']')
        DAMPvals[var_name] = {
            'units':   handle['units_'+var_name].data, #'input':   handle['input_'+var_name+'_mean'],
            'prior':   handle['prior_'+var_name+'_mean'],
            'recon':   handle['recon_'+var_name+'_mean'],
            'priorEns':handle['prior_'+var_name+'_ens'],
            'reconEns':handle['recon_'+var_name+'_ens'],
            'kalman'  :handle['kalman_'+var_name]
        }
    #Proxy
    #Load Proxy Data
    DAMPproxy={}
    for key in ['proxies_assimilated','proxy_metadata','proxy_values','proxy_resolutions','ages','proxyprior_mean','proxyrecon_mean']: 
        DAMPproxy[key]=handle[key]
    #Reshape Proxy data
    DAMPproxy['values_binned']     = np.transpose(DAMPproxy['proxy_values'].data)
    DAMPproxy['resolution_binned'] = np.transpose(DAMPproxy['proxy_resolutions'].data)
    #
    handle.close()
    return(options,DAMPvals,DAMPproxy)
            

def getLocVals(dataArray,lat,lon,locs=[1,2]):
    lati = np.argmin(np.abs(dataArray.lat.data-lat))
    loni = np.argmin(np.abs(dataArray.lon.data-lon))
    if   locs == [0,1]: out = dataArray[lati,loni]
    elif locs == [1,2]: out = dataArray[:,lati,loni]
    elif locs == [2,3]: out =  dataArray[:,:,lati,loni]
    return(out)
    
def plotBaseMap(ax,proj,lims):
    ax.spines['geo'].set_edgecolor('black')
    ax.coastlines(lw=0.3,edgecolor='grey')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.1, color='k', alpha=0.4, linestyle=(0,(5,10)))
    try:  
        ax.set_extent(lims) 
        if (lims[2] > 0): ax.add_feature(cfeature.STATES, edgecolor='grey', linewidth=0.15,zorder=1)
        ax.add_feature(cfeature.BORDERS, edgecolor='grey', linewidth=0.15,zorder=1)
    except: 
        ax.set_global()
        ax.add_feature(cfeature.BORDERS, edgecolor='grey', linewidth=0.15,zorder=1)


def calcSkill(method,x0,x1):
    ii = np.where((np.isfinite(x0)) & (np.isfinite(x1)))[0]
    if method == 'Corr':
        out = scipy.stats.pearsonr(x0[ii],x1[ii])[0]
    elif method == 'CE':
        out = 1-(np.sum((x1[ii]-x0[ii])**2)/np.sum((x1[ii]-np.mean(x1[ii]))**2))
    return(out)
    


#%%
def plotDAMPsummary(dampVals,proxy_info,var_key,options,times=[12,6],bs=1000,PSMkeys='',dampVars='',title='',proj=ccrs.Robinson(),lims=False):
    #%%
    plt.figure(figsize=(len(times)*3.5,(len(var_key)*2+2)),dpi=400)
    gs = gridspec.GridSpec(len(var_key)+1,(len(times)+1))
    for row,var_name in enumerate(var_key):
        print(var_name+': '+dampVals[var_name]['units'])
        #Load Data
        damp = dampVals[var_name]
        #
        #Plot Timeseries#########################
        ax = plt.subplot(gs[row,0]) 
        keys,colors = ['prior','trace','recon'],['grey','indigo','teal']
        for i,key in enumerate(keys):
            vals = damp[key].weighted(np.cos(np.deg2rad(damp[key].lat))).mean(("lon", "lat"))
            ax.plot(vals.ages,vals,c=colors[i],lw=1,label=key[:5])
            if key+'Ens' in damp.keys():
                valsEns = damp[key+'Ens'].weighted(np.cos(np.deg2rad(damp[key].lat))).mean(("lon", "lat"))
                ax.fill_between(valsEns.ages, valsEns.quantile(0,dim='ens_selected'), valsEns.quantile(1,dim='ens_selected'), facecolor=colors[i], alpha=0.3)
                ax.fill_between(valsEns.ages, valsEns.quantile(0.2,dim='ens_selected'), valsEns.quantile(0.8,dim='ens_selected'), facecolor=colors[i], alpha=0.7)
        #Timeseries Plot properties
        ax.set_xlim(np.quantile(damp[key]['ages'],[0,1])); ax.invert_xaxis()
        ax.xaxis.set_ticks([*range(0,22000,3000)]); ax.set_xticklabels(['' for x in range(0,22,3)])
        ax.set_ylabel(var_name+': '+dampVals[var_name]['units'],fontsize='x-small')
        ax.legend(fontsize='xx-small')
        #
        #Map Reconstruction#########################
        vals0 = damp['recon'].sel(ages=slice(0-bs/2,0+bs/2)).mean(dim='ages')
        v = np.nanquantile(np.abs(damp['recon']-vals0),0.9) 
        levs = np.linspace(-v,v,13)
        levs -= np.median(np.diff(levs))/2
        levs = np.append(levs,-levs[0])#
        #Plot Timee slice Maps
        for i,t in enumerate(times):
            valsT = damp['recon'].sel(ages=slice(t*1000-bs/2,t*1000+bs/2)).mean(dim='ages')
            ax = plt.subplot(gs[row,1+i],projection=proj) 
            plotBaseMap(ax,proj,lims)
            data_cyclic,lon_cyclic = cutil.add_cyclic_point(valsT-vals0,coord=valsT.lon.data)
            p=ax.contourf(lon_cyclic,valsT.lat,data_cyclic,levels=levs,cmap=cm.get_cmap(dampVars[var_name]['cmap'], 21),extend='both',transform=ccrs.PlateCarree())
            cbar = plt.colorbar(p,orientation='horizontal', extend='both',ticks=list(np.round(np.linspace(-v,v,4),1)),cax=inset_axes(ax,width="80%",height="10%",bbox_to_anchor=(0,-0.2,1,1),bbox_transform=ax.transAxes, loc="lower center"))
            if row==0:ax.title.set_text(str(int(t))+'-0 ka (+/-'+str((bs/2)/1000)+' ka)')
    #Plot Proxies#########################
    row+=1
    #Plot proxy density f(time)
    ax0 = plt.subplot(gs[row,0])
    ax0.set_xlim(np.quantile(damp[key]['ages'],[0,1])),ax0.set_xlabel("Age (ka)"); ax0.invert_xaxis(); ax0.xaxis.set_ticks([*range(0,22000,3000)]); ax0.set_xticklabels([*range(0,22,3)])
    ax0.set_ylabel('Count'); ax0.set_ylim(0,len(proxy_info['lats'])*1.1)
    #Plot Proxies Used
    ax = plt.subplot(gs[row,1],projection=proj) 
    plotBaseMap(ax,proj,lims)
    ax0.set_title('Proxies Used')
    #
    ages=proxy_info['proxies_assimilated'].ages
    vals=ages*0
    for i,PSM in enumerate(np.unique(proxy_info['proxyPSM'])):
        #Plot Proxies Used
        idx=(proxy_info['proxyPSM']==PSM) & (proxy_info['proxies_assimilated'].sum(axis=0)>0)
        valsNew = proxy_info['proxies_assimilated'][:,proxy_info['proxyPSM']==PSM].sum(dim='proxy')
        if i == 0: ax0.bar(ages, valsNew,              color=PSMkeys[PSM]['c'], width=np.diff(ages)[0],label=PSM.split('_')[-1]+' ('+str(int(sum(idx)))+')')
        else:      ax0.bar(ages, valsNew, bottom=vals, color=PSMkeys[PSM]['c'], width=np.diff(ages)[0],label=PSM.split('_')[-1]+' ('+str(int(sum(idx)))+')')
        vals+=valsNew
        ax.scatter(proxy_info['lons'][idx],proxy_info['lats'][idx],s=5,alpha=0.7,lw=0.3,c=PSMkeys[PSM]['c'],marker=PSMkeys[PSM]['m'],ec='k',label=PSM.split('_')[-1]+' ('+str(int(sum(idx)))+')',transform=ccrs.PlateCarree())
    ax0.legend(loc='lower right',fontsize='x-small')
    #
    #Plot Proxies Withheld #########################
    for i,method in enumerate(['Corr','CE']):
        ax = plt.subplot(gs[row,(i+2)],projection=proj) 
        plotBaseMap(ax,proj,lims)
        idx=np.where((proxy_info['proxyPSM']==PSM) & (proxy_info['proxies_assimilated'].sum(axis=0)>0) & (np.sum(np.isfinite(proxy_info['proxy_values']),axis=0)>0) & (np.sum(np.isfinite(proxy_info['proxyprior_mean']),axis=0)>0))[0]            
        #
        xs = proxy_info['proxyrecon_mean'][:,idx]
        x0s =  [getLocVals(dampVals[proxy_info['proxyPSM'][x].split('_')[-1]]['trace'],proxy_info['lats'][x],proxy_info['lons'][x]) for x in idx]
        x0s = np.transpose(np.array(x0s))
        ys = proxy_info['proxy_values'][:,idx]
        #
        agei = np.where(xs.ages<12000)[0]
        #
        skills  = [calcSkill(method,xs[agei,x],ys[agei,x]) for x in range(len(idx))]
        skills0 = [calcSkill(method,x0s[agei,x],ys[agei,x]) for x in range(len(idx))]
        skillDiff = [skills[x]-skills0[x] for x in range(len(skills))]
        #
        v = [-1,1]
        p = ax.scatter(proxy_info['lons'][idx],proxy_info['lats'][idx],c=np.array(skills),vmin=v[0],vmax=v[1],cmap='Spectral_r',
                        s=20,alpha=0.7,lw=0.6,ec='k',marker=PSMkeys[PSM]['m'],transform=ccrs.PlateCarree())
        cbar = plt.colorbar(p,orientation='horizontal',ticks=list(np.linspace(v[0],v[1],3)),cax=inset_axes(ax,width="80%",height="10%",bbox_to_anchor=(0,-0.2,1,1),bbox_transform=ax.transAxes, loc="lower center"))
        try: cbar.set_label('Withheld '+method+' \n('+str(np.round(np.nanmean(skills),2))+')'+'('+str(np.round(np.nanmean(skillDiff),2))+')')
        except:  cbar.set_label('withheld '+method)
    #%% #
    plt.suptitle(title)
    plt.tight_layout()
    return(plt,skills)

#%%


#A 

def plotDAMPinnovation(var_name,dampVals,proxy_info,times=[18,12,6],proxybin=False,title='Update',PSMkeys={},dampVars='',proj=ccrs.Robinson(),lims=False):
    #scale to reduce color bar
    s=3
    ncol=len(times)
    proxies = (proxy_info['proxyPSM']=='get_'+var_name) & (np.sum(np.isfinite(proxy_info['proxyprior_mean']),axis=0) > 0)
    #Data for PSM
    prior,recon = dampVals[var_name]['prior']*1, dampVals[var_name]['recon']*1
    prior -= dampVals[var_name]['prior'].sel(ages=slice(0,12000)).mean(dim='ages')
    recon -= dampVals[var_name]['recon'].sel(ages=slice(0,12000)).mean(dim='ages')
    innov = recon-prior
    #Set up plot
    plt.figure(figsize=(ncol*3,len(times)*2.1),dpi=400)
    gs = gridspec.GridSpec(len(times)*s+1,3)
    #Standard Colorbar Range
    v=round(np.nanquantile(np.abs(recon),0.85),2)
    levs = np.linspace(-v,v,13)
    levs -= np.median(np.diff(levs))/2
    levs = np.append(levs,-levs[0])
    cmap = cm.get_cmap(dampVars[var_name]['cmap'], 31)
    #Plot
    for row,t in enumerate(times):
        t_i = np.argmin(np.abs(prior['ages'].data-t*1000))
        for col,name in enumerate(['prior','innov','recon']):
            #Get Data to plot
            if name == 'recon':
                vals, text =  recon[t_i],'Reconstruction & Proxies Withheld'
                proxyi = (proxies) & (proxy_info['proxies_assimilated'][t_i]==0)
            else:
                proxyi = (proxies) & (proxy_info['proxies_assimilated'][t_i]==1) 
                if   name == 'prior': vals,text = prior[t_i],'Prior & Proxies Used'
                elif name == 'innov': vals,text = innov[t_i],'Innovation'
            #
            proxyi=np.where(proxyi & np.isfinite(proxy_info['proxyprior_mean'][t_i]))[0]
            proxyvals = proxy_info['proxy_values'][:,proxyi]
            proxyvals = proxyvals[t_i]-proxyvals.sel(ages=slice(0,12000)).mean(dim='ages')
            #Calc diff between proxy and prior
            if name == 'innov': 
                proxyvals = [float((proxyvals[i] - getLocVals(prior[t_i,:,:],proxy_info['lats'][x],proxy_info['lons'][x],[0,1])).data) for i,x in enumerate(proxyi)]
            #Set up Map
            ax = plt.subplot(gs[(row*s):(row*s+s),col],projection=proj) 
            plotBaseMap(ax,proj,lims)
            #Plot gridded data
            data_cyclic,lon_cyclic = cutil.add_cyclic_point(vals,coord=vals.lon.data)
            p=ax.contourf(lon_cyclic,vals.lat,data_cyclic,levels=levs,extend='both',cmap=cmap,transform=ccrs.PlateCarree())
            #Plot proxies
            ii=np.isfinite(proxyvals)
            if sum(ii)>0: 
                lats,lons = proxy_info['lats'][proxyi][ii],proxy_info['lons'][proxyi][ii]
                ax.scatter(np.array(lons),np.array(lats),c=np.array(proxyvals)[ii],
                           vmin=-v,vmax=v, cmap=cmap, s=20,ec='k',lw=0.2,transform=ccrs.PlateCarree())
            #Label Plot
            if row==0: ax.set_title(text+'\n'+str(t)+'ka',fontsize='small')
            else:      ax.set_title(          str(t)+'ka',fontsize='small')
    #Colorbar
    ax = plt.subplot(gs[(row*s+s),:])
    ax.set_axis_off()
    cbar = plt.colorbar(p,orientation='horizontal',ticks=[-v,-v/2,0,v,v],extend='both',fraction=1,aspect=100)#,cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
    cbar.set_label(dampVals[var_name]['units'][0])
    cbar.ax.tick_params(labelsize='x-small') 
    plt.suptitle(title+'\n'+var_name)
    plt.tight_layout()
    return(plt)



#%%

def plotKalmanGain(dampVals,proxy_info,var_name,ncol=5,latbounds=[[90,30],[30,0],[0,-90]],cmap='BrBG',title=''):
    #scale to reduce color bar
    s=3
    plt.figure(figsize=(ncol*3+1,len(latbounds)*2),dpi=400)
    gs = gridspec.GridSpec(len(latbounds)*s+1,ncol)
    v = np.nanquantile(np.abs(dampVals[var_name]['kalman'][0]),0.999)
    #Plot
    for row,lats in enumerate(latbounds):
        #Pick sites within lat range
        idx = np.where((proxy_info['lats'] >= lats[1]) & (proxy_info['lats'] <= lats[0]) & (proxy_info['proxies_assimilated'][0]>0))[0]
        #Pick evenly spaced lons within latbounds
        try: idx = [idx[y] for y in [proxy_info['lons'][idx].argsort()[int(x)] for x in np.linspace(0,len(idx)-1,ncol)]]
        except: idx =np.array([0])  
        #Plot
        for col,i in enumerate(idx):
            ax = plt.subplot(gs[(row*s):(row*s+s),col],projection=ccrs.Robinson(central_longitude=proxy_info['lons'][i])) 
            ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(lw=0.2)
            #
            vals = dampVals[var_name]['kalman'].median(dim='ages')[i]
            #v = np.nanmax(np.abs(vals))
            p = ax.pcolormesh(vals.lon,vals.lat,vals,cmap=cmap,vmin=-v,vmax=v,transform=ccrs.PlateCarree())
            ax.scatter(proxy_info['lons'][i],proxy_info['lats'][i],transform=ccrs.PlateCarree(),ec='k',lw=1,color='none',s=40)
            ax.scatter(proxy_info['lons'][i],proxy_info['lats'][i],transform=ccrs.PlateCarree(),color='k',s=0.05)
            #
    ax = plt.subplot(gs[(row*s+s),:])
    ax.set_axis_off()
    cbar = plt.colorbar(p,orientation='horizontal',ticks=[-v,-v/2,0,v/2,v],extend='both',fraction=1,aspect=100)#,cax=inset_axes(ax,width="7%",height="80%",bbox_to_anchor=(0,0,1,1),bbox_transform=ax.transAxes, loc="center left"))
    cbar.set_label(dampVals[var_name]['units'])
    cbar.ax.tick_params(labelsize='x-small') 
    plt.suptitle(title+'\n'+var_name)
    plt.tight_layout()
    return(plt)





