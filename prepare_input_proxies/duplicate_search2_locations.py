#==============================================================================
# This script reads in proxy data. For each location with multiple records, it
# makes one plot, so users can conduct a visual comparison.
#    author: Michael Erb
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import rdata
import geopy.distance

plt.style.use('ggplot')
save_instead_of_plot = False

# Settings (distance in km)
distance_threshhold = 10
#interp_selected = 'temperature'; interp_to_find = ['temperature']
interp_selected = 'precip';      interp_to_find = ['precipitation','effectivePrecipitation']


#%% LOAD DATA

da_dir     = 'P:/data_paleoclimate/data_assimilation/proxies/ecoclimate/'
figure_dir = 'C:/Users/erbm/Documents/GitHub/Holocene-code/prepare_input_proxies/figures/'

# Load the proxy data
all_ts = rdata.read_rds(da_dir+'ecoclimate_selected_ts_2026-07-16.rds')
print('Number of records total:',len(all_ts))


#%% GET PROXIES OF SELECTED SEASON

filtered_ts = []
for i in range(len(all_ts)):
    proxy_interp = all_ts[i]['interpretation1_variable'][0]
    print(proxy_interp)
    if proxy_interp in interp_to_find:
        filtered_ts.append(all_ts[i])


#%% COMPUTE DISTANCES

n_proxies = len(filtered_ts)
distances_km = np.zeros((n_proxies,n_proxies)); distances_km[:] = np.nan
counter = 0
i=0; j=1
for i in range(n_proxies):
    proxy_lat_1 = filtered_ts[i]['geo_latitude'][0]
    proxy_lon_1 = filtered_ts[i]['geo_longitude'][0]
    distances_km[i,i] = 0
    for j in range(i+1,n_proxies):
        proxy_lat_2 = filtered_ts[j]['geo_latitude'][0]
        proxy_lon_2 = filtered_ts[j]['geo_longitude'][0]
        #
        # Calculate distance
        distances_km[i,j] = geopy.distance.great_circle((proxy_lat_1,proxy_lon_1),(proxy_lat_2,proxy_lon_2)).km


#%% FIND GROUPS OF NEARBY PROXIES

groups = {}
group_key = 0
for i in range(n_proxies):
    # Find other proxies close to the selected proxy
    distances_to_selected_proxy = distances_km[i,:]
    ind_within_distance = np.where((distances_to_selected_proxy <= distance_threshhold))[0]
    # If there are more than 1 proxies found, create a group
    if len(ind_within_distance) > 1:
        groups[group_key] = ind_within_distance
        group_key += 1
    # Set selected distances to nan, so selected proxies don't appear in any more groups
    distances_km[:,ind_within_distance] = np.nan


#%% LOOP THROUGH PROXIES, COMPUTING ALL DISTANCES

group_keys_all = list(groups.keys())
group_key = group_keys_all[0]
for group_key in group_keys_all:
    #
    # Plot observations
    f, ax1 = plt.subplots(1,1,figsize=(10,6))
    for i in groups[group_key]:
        proxy_values           = np.array(filtered_ts[i]['paleoData_values']).astype(float)
        proxy_ages             = np.array(filtered_ts[i]['age']).astype(float)
        proxy_lat              = filtered_ts[i]['geo_latitude'][0]
        proxy_lon              = filtered_ts[i]['geo_longitude'][0]
        try:    proxy_variable = str(filtered_ts[i]['paleoData_variableName'][0])
        except: proxy_variable = 'missing'
        try:    proxy_archive = str(filtered_ts[i]['archiveType'][0])
        except: proxy_archive = 'missing'
        try:    proxy_proxy = str(filtered_ts[i]['paleoData_proxy'][0])
        except: proxy_proxy = 'missing'
        try:    proxy_units   = str(filtered_ts[i]['paleoData_units'][0])
        except: proxy_units   = 'missing'
        try:    proxy_season   = str(filtered_ts[i]['interpretation1_seasonality'][0])
        except: proxy_season   = 'missing'
        try:    proxy_datasetname = str(filtered_ts[i]['dataSetName'][0])
        except: proxy_datasetname = 'missing'
        try:    proxy_tsid = str(filtered_ts[i]['paleoData_TSid'][0])
        except: proxy_tsid = 'missing'
        proxy_id = proxy_variable+' - '+proxy_archive+' - '+proxy_proxy+' - '+proxy_units+' - '+proxy_season+' - '+proxy_datasetname+' - '+proxy_tsid
        ax1.plot(proxy_ages,proxy_values,'o-',markersize=5,linestyle='None',label=proxy_id)

    ax1.legend(loc='upper center',title="variable - archive - proxy - units - seasonality - datasetname - tsid",bbox_to_anchor=(0,0,1,-.1))
    ax1.set_ylabel('Value')
    ax1.set_xlabel('Age B.P')
    ax1.set_xlim(21000,0)
    ax1.set_title(interp_selected+' records near '+str('%1.1f' % proxy_lat)+'$^\circ$N, '+str('%1.1f' % proxy_lon)+'$^\circ$E')
    #
    if save_instead_of_plot:
        lat1_txt = str('%1.5f' % ((-1*proxy_lat)+90))  # This is to order the records by latitude, from north to south
        plt.savefig(figure_dir+interp_selected+'_group_'+str(group_key)+'_'+lat1_txt+'_ts.png',dpi=150,format='png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
