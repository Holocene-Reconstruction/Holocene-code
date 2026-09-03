#==============================================================================
# This script reads in proxy data and searches for duplicates. Potential
# duplicates are put into three sets:
#  - Set 1: paleodata values are the same                           - probably duplicates
#  - Set 2: values are different, but datasetname is also different - might be duplicates
#  - Set 3: values are different, but datasetname is the same       - probably not duplicates (e.g., different seasons)
#    author: Michael Erb
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import rdata
import geopy.distance
import pandas as pd

plt.style.use('ggplot')
save_instead_of_plot = False

# Settings
n_points_necessary = 10
corr_threshold     = 0.99


#%% LOAD DATA

da_dir     = 'P:/data_paleoclimate/data_assimilation/proxies/ecoclimate/'
figure_dir = 'C:/Users/erbm/Documents/GitHub/Holocene-code/prepare_input_proxies/figures/'

# Load the proxy data
filtered_ts = rdata.read_rds(da_dir+'ecoclimate_selected_ts_2026-07-16.rds')
print('Number of records total:',len(filtered_ts))

# List all metadata keys
keys_all = []
for i in range(len(filtered_ts)): keys_all.extend(list(filtered_ts[0].keys()))
print(np.unique(keys_all))


#%% BIN DATA

# Set up age bounds for regridding
age_bounds = np.arange(0,22001,100)
age_centers = (age_bounds[1:] + age_bounds[:-1])/2
n_ages    = len(age_centers)
n_proxies = len(filtered_ts)

# Set up an array
proxies_binned = np.zeros((n_proxies,n_ages)); proxies_binned[:] = np.nan

# Bin all of the records at 100 year resolution
i=0
for i in range(n_proxies):
    #
    # Get proxy data
    proxy_values = np.array(filtered_ts[i]['paleoData_values']).astype(float)
    proxy_ages   = np.array(filtered_ts[i]['age']).astype(float)
    #
    # Bin the proxy data
    for j in range(n_ages):
        ind_selected = np.where((proxy_ages >= age_bounds[j]) & (proxy_ages < age_bounds[j+1]))[0]
        proxies_binned[i,j] = np.nanmean(proxy_values[ind_selected])
    #
    # Check for records with no data in the selected time window
    if np.isnan(np.nanmean(proxies_binned[i,:])): print('All nans for record',i,filtered_ts[i]['paleoData_TSid'])

# Calculate all correlations
correlations = pd.DataFrame(np.transpose(proxies_binned)).corr(method='pearson',min_periods=n_points_necessary).values
#print(correlations.shape)


#%% LOOP THROUGH PROXIES, MAKING FIGURES FOR POSSIBLE DUPLICATES

# Create a dataframe to store info in
possible_duplicates_df = pd.DataFrame(columns=["counter","set","datasetname1","datasetname2","tsid1","tsid2","variable1","variable2","season1","season2","lat1","lon1","correlation"])

# Select the keys to show on the output figures
keys_to_print = ['dataSetName','paleoData_TSid','age','paleoData_values','geo_latitude','geo_longitude','archiveType','paleoData_variableName',
                 'interpretation1_variable','interpretation1_seasonalityGeneral','interpretation1_seasonality',
                 'paleoData_useInGlobalTemperatureAnalysis','changelog','createdBy','geo_siteName','lipdVersion',
                 'paleoData_QCCertification','paleoData_QCnotes','paleoData_inCompilationBeta1_compilationName','paleoData_proxyGeneral','paleoData_proxyDetail',
                 'pub1_author','pub1_citKey','pub1_doi','pub1_journal','pub1_year','ageUnits',
                 "paleoData_isPrimary","paleoData_primaryTimeseries"]

distances_km = np.zeros((n_proxies,n_proxies)); distances_km[:] = np.nan
counter = 0
i=0; j=1
for i in range(n_proxies):
    #
    # Load proxy 1
    proxy_lat_1 = filtered_ts[i]['geo_latitude'][0]
    proxy_lon_1 = filtered_ts[i]['geo_longitude'][0]
    #
    for j in range(i+1,n_proxies):
        #
        # Load proxy 2
        proxy_lat_2 = filtered_ts[j]['geo_latitude'][0]
        proxy_lon_2 = filtered_ts[j]['geo_longitude'][0]
        #
        # Calculate distance
        distances_km[i,j] = geopy.distance.great_circle((proxy_lat_1,proxy_lon_1),(proxy_lat_2,proxy_lon_2)).km
        #
        # If values are similar enough, make a figure
        if (np.abs(correlations[i,j]) >= corr_threshold) & (np.abs(proxy_lat_1-proxy_lat_2) <= 1) & (np.abs(proxy_lon_1-proxy_lon_2) <= 1):
            #
            # Print some output
            #print('Correlation = '+str('%1.5f' % correlations[i,j]))
            proxy_id_1 = filtered_ts[i]['dataSetName'][0]+' - '+filtered_ts[i]['paleoData_TSid'][0]
            proxy_id_2 = filtered_ts[j]['dataSetName'][0]+' - '+filtered_ts[j]['paleoData_TSid'][0]
            data1 = np.array(filtered_ts[i]['paleoData_values'])
            data2 = np.array(filtered_ts[j]['paleoData_values'])
            ages1 = np.array(filtered_ts[i]['age'])
            ages2 = np.array(filtered_ts[j]['age'])
            try:    variable1 = str(filtered_ts[i]['paleoData_variableName'][0])
            except: variable1 = ''
            try:    variable2 = str(filtered_ts[j]['paleoData_variableName'][0])
            except: variable2 = ''
            try:    season1 = str(filtered_ts[i]['interpretation1_seasonality'][0])
            except: season1 = ''
            try:    season2 = str(filtered_ts[j]['interpretation1_seasonality'][0])
            except: season2 = ''
            #
            # Sort the ages, then reassign them to dataframes
            ind_sorted1 = np.argsort(ages1)
            data1 = data1[ind_sorted1]
            ages1 = ages1[ind_sorted1]
            ind_sorted2 = np.argsort(ages2)
            data2 = data2[ind_sorted2]
            ages2 = ages2[ind_sorted2]
            filtered_ts[i]['paleoData_values'] = data1
            filtered_ts[j]['paleoData_values'] = data2
            filtered_ts[i]['age'] = ages1
            filtered_ts[j]['age'] = ages2
            #
            # Figure out which set the possible match belongs to
            if   np.array_equal(data1,data2):                                    set_txt = 'set1_values_same'
            elif filtered_ts[i]['dataSetName'] != filtered_ts[j]['dataSetName']: set_txt = 'set2_datasetname_different'
            else:                                                                set_txt = 'set3_datasetname_same'
            #
            # Save some metadata
            new_df = pd.DataFrame({"counter": [counter],
                                   "set": [set_txt],
                                   "datasetname1": [filtered_ts[i]['dataSetName'][0]],
                                   "datasetname2": [filtered_ts[j]['dataSetName'][0]],
                                   "tsid1": [filtered_ts[i]['paleoData_TSid'][0]],
                                   "tsid2": [filtered_ts[j]['paleoData_TSid'][0]],
                                   "variable1": [variable1],
                                   "variable2": [variable2],
                                   "season1": [season1],
                                   "season2": [season2],
                                   "lat1": [proxy_lat_1],
                                   "lon1": [proxy_lon_1],
                                   "correlation": [correlations[i,j]]})
            possible_duplicates_df = pd.concat([possible_duplicates_df,new_df], ignore_index=True)
            #
            # Get the binned proxy values
            proxy_data_1 = proxies_binned[i,:]
            proxy_data_2 = proxies_binned[j,:]
            valid_data = np.isfinite(proxy_data_1+proxy_data_2)
            #
            # Plot observations
            f, ax = plt.subplots(1,1,figsize=(20,6),sharex=True)
            ax.plot(age_centers,proxy_data_1,color='k',   marker='o',markersize=20,linestyle='None',label=proxy_id_1+' ('+str('%1.1f' % proxy_lat_1)+'$^\circ$N, '+str('%1.1f' % proxy_lon_1)+'$^\circ$E)')
            ax.plot(age_centers,proxy_data_2,color='gray',marker='o',markersize=10,linestyle='None',label=proxy_id_2+' ('+str('%1.1f' % proxy_lat_2)+'$^\circ$N, '+str('%1.1f' % proxy_lon_2)+'$^\circ$E)')
            ax.legend()
            ax.set_ylabel('Value')
            ax.set_xlabel('Age B.P')
            ax.set_xlim(22000,0)
            ax.set_title('Possible duplicates. Correlation='+str('%1.5f' % correlations[i,j])+'. N_points_overlap='+str(sum(valid_data)))
            #
            # Metadata display parameters
            fntsize        = 14
            offsetscale    = 0.07
            initial_offset = -.37
            string_limit   = 40
            #
            # Print metadata
            plt.text(.5, -0.19,'Metadata (differences are highlighted; only first '+str(string_limit)+' characters are displayed)',transform=ax.transAxes,ha='center',fontsize=22)
            plt.text(0,  -0.3,'Key',              transform=ax.transAxes,fontsize=22)
            plt.text(.3, -0.3,'Record 1 (black)', transform=ax.transAxes,fontsize=22)
            plt.text(.65,-0.3,'Record 2 (gray)',  transform=ax.transAxes,fontsize=22)
            #
            for keynum,key in enumerate(keys_to_print):
                try:    metadata_field1 = str(filtered_ts[i][key])
                except: metadata_field1 = ''
                try:    metadata_field2 = str(filtered_ts[j][key])
                except: metadata_field2 = ''
                if metadata_field1 != metadata_field2:
                    plt.text(0,initial_offset-offsetscale*keynum,' '*250,transform=ax.transAxes,fontsize=fntsize,bbox={'facecolor':'y','alpha':.8,'pad':5})
                #
                plt.text(0,  initial_offset-offsetscale*keynum,key,                            transform=ax.transAxes,fontsize=fntsize)
                plt.text(.3, initial_offset-offsetscale*keynum,metadata_field1[0:string_limit],transform=ax.transAxes,fontsize=fntsize)
                plt.text(.65,initial_offset-offsetscale*keynum,metadata_field2[0:string_limit],transform=ax.transAxes,fontsize=fntsize)
            #
            if save_instead_of_plot:
                lat1_txt = str('%1.5f' % ((-1*filtered_ts[i]['geo_latitude'])+90))  # This is to order the records by latitude, from north to south
                plt.savefig(figure_dir+set_txt+'_possible_duplicate_lat_'+lat1_txt+'_'+str(counter).zfill(3)+'_ts.png',dpi=150,format='png',bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            #
            counter += 1

# Sort dataframe and write to excel file
possible_duplicates_df = possible_duplicates_df.sort_values(by=["set","lat1","counter"])
possible_duplicates_df.to_excel(figure_dir+"metadata_possible_duplicates.xlsx")
