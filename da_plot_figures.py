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


#%%
def plotDAMPsummary(dampVals,proxy_info,var_key,times=[12,6],bs=1000,PSMkeys='',dampVars='',title=''):
    #%%
    plt.figure(figsize=(len(times)*3.5,(len(var_key)*2+2)),dpi=400)
    gs = gridspec.GridSpec(len(var_key)+1,(len(times)+1))
    for row,var_name in enumerate(var_key):
        print(var_name+': '+dampVals[var_name]['units'])
        #titlabel
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
        ax.set_xlim(np.quantile(damp[key]['ages'],[0,1])); ax.invert_xaxis()#ax.set_xlabel("Age (ka)"); 
        ax.set_ylabel(var_name+': '+dampVals[var_name]['units'])
        ax.xaxis.set_ticks([*range(0,22000,3000)]); ax.set_xticklabels(['' for x in range(0,22,3)])
        ax.legend(fontsize='x-small')
        #
        #Map Reconstruction#########################
        vals0 = damp['recon'].sel(ages=slice(0-bs/2,0+bs/2)).mean(dim='ages')
        v = np.nanquantile(np.abs(damp['recon']-vals0),0.9) 
        levs = np.linspace(-v,v,13)
        levs -= np.median(np.diff(levs))/2
        levs = np.append(levs,-levs[0])#
        #Plot Timeselice Maps
        for i,t in enumerate(times):
            valsT = damp['recon'].sel(ages=slice(t*1000-bs/2,t*1000+bs/2)).mean(dim='ages')
            #
            ax = plt.subplot(gs[row,1+i],projection=ccrs.Robinson()) 
            ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(lw=0.2)
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.2, color='k', alpha=0.4, linestyle=(0,(5,10)))
            #
            data_cyclic,lon_cyclic = cutil.add_cyclic_point(valsT,coord=valsT.lon.data)
            p=ax.contourf(lon_cyclic,valsT.lat,data_cyclic,levels=levs,cmap=cm.get_cmap(dampVars[var_name]['cmap'], 21),extend='both',transform=ccrs.PlateCarree())
            cbar = plt.colorbar(p,orientation='horizontal', extend='both',ticks=list(np.round(np.linspace(-v,v,4),1)),cax=inset_axes(ax,width="80%",height="10%",bbox_to_anchor=(0,-0.2,1,1),bbox_transform=ax.transAxes, loc="lower center"))
            if row==0:ax.title.set_text(str(int(t))+'-0 ka (+/-'+str((bs/2)/1000)+' ka)')
            #if (var_name == 'LakeStatus') & (proxy_info):
             #   proxies = proxy_info['proxy_values'].sel(ages=slice(t*1000-bs/2,t*1000+bs/2)).mean(dim='ages') - proxy_info['proxy_values'].sel(ages=slice(0*1000-bs/2,0*1000+bs/2)).mean(dim='ages')
               # ax.scatter(proxy_info['lons'][np.where(np.isfinite(proxies))[0]],proxy_info['lats'][np.where(np.isfinite(proxies))[0]],c=proxies[np.where(np.isfinite(proxies))[0]],s=20,ec='k',vmin=-v,vmax=v,cmap=cm.get_cmap(dampVars[var_name]['cmap'], 21),transform=ccrs.PlateCarree())
    #Plot Proxies#########################
    row+=1
    #Plot proxy density f(time)
    ax0 = plt.subplot(gs[row,0])
    #ax1.title.set_text('Proxy density over time')
    ax0.set_xlim(np.quantile(damp[key]['ages'],[0,1])),ax0.set_xlabel("Age (ka)"); ax0.invert_xaxis(); ax0.xaxis.set_ticks([*range(0,22000,3000)]); ax0.set_xticklabels([*range(0,22,3)])
    ax0.set_ylabel('Count'); ax0.set_ylim(0,len(proxy_info['lats'])*1.1)
    #Plot Proxies Used
    ax1 = plt.subplot(gs[row,1],projection=ccrs.Robinson()) 
    ax1.spines['geo'].set_edgecolor('black'); ax1.set_global(); ax1.coastlines(linewidth=0.3)
    ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.2, color='k', alpha=0.4, linestyle=(0,(5,10)))
    ax0.set_title('Proxies Used')
    #
    ages=proxy_info['proxies_assimilated'].ages
    vals=ages*0
    for i,PSM in enumerate(np.unique(proxy_info['proxyPSM'])):
        #Plot Proxies Used
        idx=(proxy_info['proxyPSM']==PSM) & (proxy_info['proxies_assimilated'].sum(axis=0)>0)
        valsNew = proxy_info['proxies_assimilated'][:,proxy_info['proxyPSM']=='get_LakeStatus'].sum(dim='proxy')
        vals+=valsNew
        if i == 0: ax0.bar(ages, valsNew,              color=PSMkeys[PSM]['c'], width=np.diff(ages)[0],label=PSM.split('_')[-1]+' ('+str(int(sum(idx)))+')')
        else:      ax0.bar(ages, valsNew, bottom=vals, color=PSMkeys[PSM]['c'], width=np.diff(ages)[0],label=PSM.split('_')[-1]+' ('+str(int(sum(idx)))+')')
        #
        ax1.scatter(proxy_info['lons'][idx],proxy_info['lats'][idx],s=5,alpha=0.7,lw=0.3,c=PSMkeys[PSM]['c'],marker=PSMkeys[PSM]['m'],ec='k',label=PSM.split('_')[-1]+' ('+str(int(sum(idx)))+')',transform=ccrs.PlateCarree())
        #
    ax0.legend(loc='lower right',fontsize='x-small')
    #
    #Plot Proxies Withheld
    for i,method in enumerate(['Correlation','CE']):
        ax = plt.subplot(gs[row,(i+2)],projection=ccrs.Robinson()) 
        ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(linewidth=0.3)
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,lw=0.2, color='k', alpha=0.4, linestyle=(0,(5,10)))
        idx=np.where((proxy_info['proxyPSM']==PSM) & (proxy_info['proxies_assimilated'].sum(axis=0)==0))[0]            
        xs = proxy_info['proxyrecon_mean'][:,idx]
        x0s = proxy_info['proxyprior_mean'][:,idx]
        ys = proxy_info['proxy_values'][:,idx]
        skills,skillDiff,idx2=[],[],[]
        for proxy in range(len(idx)):
            if method == 'Correlation': 
                x,y,x0 = xs[:,proxy], ys[:,proxy], x0s[:,proxy]
                ii = (np.isfinite(x)) & (np.isfinite(y)) & (np.isfinite(x0))
                if sum(ii) > 0:
                    skill  = scipy.stats.pearsonr(x[ii],y[ii])[0]; 
                    skill0 = scipy.stats.pearsonr(x0[ii],y[ii])[0]; 
                    skills.append(skill);idx2.append(proxy);skillDiff.append(skill-skill0)
                v=[-1,1]
            elif method == 'CE':
                x,y,x0 = xs[:,proxy], ys[:,proxy], x0s[:,proxy]
                ii = (np.isfinite(x)) & (np.isfinite(y)) &  (np.isfinite(x0))
                if sum(ii) > 0:
                    skill  = 1-(np.sum((y[ii]-x[ii])**2)/np.sum((y[ii]-np.mean(y[ii]))**2));
                    skill0 = 1-(np.sum((y[ii]-x0[ii])**2)/np.sum((y[ii]-np.mean(y[ii]))**2));
                    skills.append(skill);idx2.append(proxy);skillDiff.append(skill-skill0)
                v=[-1,1]
            #
        p = ax.scatter(proxy_info['lons'][idx][idx2],proxy_info['lats'][idx][idx2],c=skills,vmin=v[0],vmax=v[1],cmap='Spectral_r',
                        s=20,alpha=0.7,lw=0.6,ec='k',marker=PSMkeys[PSM]['m'],transform=ccrs.PlateCarree())

        cbar = plt.colorbar(p,orientation='horizontal',ticks=list(np.linspace(v[0],v[1],3)),cax=inset_axes(ax,width="80%",height="10%",bbox_to_anchor=(0,-0.2,1,1),bbox_transform=ax.transAxes, loc="lower center"))
        try: cbar.set_label('Withheld '+method+' ('+str(np.round(np.nanmedian(skills),2))+')'+'('+str(np.round(np.nanmedian(skillDiff),2))+')')
        except:  cbar.set_label('withheld '+method)
    #%% #
    plt.suptitle(title)
    plt.tight_layout()
    return(plt,skills)

#%%


#A 

def plotDAMPinnovation(var_name,dampVals,proxy_info,PSM,times=[18,12,6,0],proxybin=False,title='Update',PSMkeys={},dampVars=''):
    #scale to reduce color bar
    s=3
    #Data for PSM
    prior,recon = dampVals[var_name]['prior'], dampVals[var_name]['recon']
    innov = recon-prior
    #Set up plot
    ncol=3
    plt.figure(figsize=(ncol*3,len(times)*2.1),dpi=400)
    gs = gridspec.GridSpec(len(times)*s+1,3)
    #Standard Colorbar Range
    v=round(np.nanquantile(np.abs(recon),0.85),2)
    levs = np.linspace(-v,v,13)
    levs -= np.median(np.diff(levs))/2
    levs = np.append(levs,-levs[0])
    #Plot
    for row,t in enumerate(times):
        for col,_ in enumerate(['prior','recon','innov']):
            t_i = np.argmin(np.abs(prior['ages'].data-t*1000))
            #Get Data to plot
            if col == 0: 
                text = 'Prior & Proxies Used'
                vals = prior[t_i]
                if PSM: 
                    proxyi = np.where((proxy_info['proxyPSM']==PSM) & (proxy_info['proxies_assimilated'][t_i]==1))[0]
                    proxyvals=proxy_info['proxy_values'][t_i][proxyi]
            elif col == 1: #Innovation
                text = 'Innovation'
                vals = innov[t_i]
                if PSM: 
                    proxyi = np.where((proxy_info['proxyPSM']==PSM) & (proxy_info['proxies_assimilated'][t_i]==1))[0]
                    proxyvals =proxy_info['proxy_values'][t_i][proxyi]
                    proxyvals-=proxy_info['proxyprior_mean'][t_i][proxyi]
            elif col == 2:
                text = 'Reconstruction & Proxies Withheld'
                vals = recon[t_i]
                if PSM: 
                    proxyi = np.where((proxy_info['proxyPSM']==PSM) & (proxy_info['proxies_assimilated'].sum(axis=0) == 0) & (np.isfinite(proxy_info['proxy_values']).sum(axis=0)>0))[0]
                    proxyvals=proxy_info['proxy_values'][t_i][proxyi]
            #Set up Map
            ax = plt.subplot(gs[(row*s):(row*s+s),col],projection=ccrs.Robinson()) 
            ax.spines['geo'].set_edgecolor('black'); ax.set_global(); ax.coastlines(lw=0.2)
            #Plot gridded data
            data_cyclic,lon_cyclic = cutil.add_cyclic_point(vals,coord=vals.lon.data)
            p=ax.contourf(lon_cyclic,vals.lat,data_cyclic,levels=levs,extend='both',cmap=cm.get_cmap(dampVars[var_name]['cmap'], 21),transform=ccrs.PlateCarree())
            #Plot proxies
            if PSM: 
                if proxybin: #Less Clutter
                    lats,lons,proxyarray = [],[],[]
                    for lat in range(-90,91,proxybin[0]):
                        for lon in range(0,360,proxybin[1]):
                            i = np.where((proxy_info['lats'][proxyi]>=lat-proxybin[0]/2) & (proxy_info['lats'][proxyi]< lat+proxybin[0]/2) &(proxy_info['lons'][proxyi]>=lon-proxybin[1]/2) & (proxy_info['lons'][proxyi]< lon+proxybin[1]/2))[0]
                            if len(i) > 0: 
                                lats.append(lat);lons.append(lon);
                                proxyarray.append(np.nanmean(proxyvals[i]))
                else: #All the Data
                    lats,lons = proxy_info['lons'][proxyi],proxy_info['lats'][proxyi]
                    proxyarray= proxyvals
                ii=np.where(np.isfinite(proxyarray))[0]
                ax.scatter(np.array(lons)[ii],np.array(lats)[ii],c=np.array(proxyarray)[ii],
                           vmin=-v,vmax=v,cmap=cm.get_cmap(PSMkeys[PSM]['cmap'], 21),s=8,ec='k',lw=0.1,transform=ccrs.PlateCarree())
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
    v = np.nanquantile(np.abs(dampVals[var_name]['kalman'][0]),0.99)
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





