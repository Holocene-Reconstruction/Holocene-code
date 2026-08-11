#==============================================================================
# General utilities
#    author: Michael Erb
#==============================================================================

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import copy

# Print a sorted list of the variable
def print_sorted_list(variable_all,title_txt,count_min=1,print_format='%55s %5s'):
    #
    # Count the number of each name
    name_words,name_counts = np.unique(variable_all,return_counts=True)
    count_sort_ind = np.argsort(-name_counts)
    name_words_sorted  = name_words[count_sort_ind]
    name_counts_sorted = name_counts[count_sort_ind]
    #
    # Print the counts
    print('\n'+title_txt)
    for i in range(len(name_counts_sorted)):
        output_txt = print_format % (name_words_sorted[i],name_counts_sorted[i])
        if name_counts_sorted[i] >= count_min: print(output_txt)


# Get all values of a selected variable
def overview(selected_ts,var_txt,count_min=1,print_format='%55s %5s'):
    #
    var_all = []
    for i in range(len(selected_ts)):
        if var_txt == 'paleoData_interpretation_0_variable':
            try:    var_value = selected_ts[i]['paleoData_interpretation'][0]['variable']
            except: var_value = 'Not given'
        elif var_txt == 'paleoData_interpretation_0_seasonality':
            try:    var_value = selected_ts[i]['paleoData_interpretation'][0]['seasonality']
            except: var_value = 'Not given'
        elif var_txt == 'paleoData_interpretation_0_seasonalityGeneral':
            try:    var_value = selected_ts[i]['paleoData_interpretation'][0]['seasonalityGeneral']
            except: var_value = 'Not given'
        else:
            #try:    var_value = selected_ts[i][var_txt]
            try:
                var_value_all = selected_ts[i][var_txt]
                if len(var_value_all) == 1: var_value = selected_ts[i][var_txt][0]
                else:                       var_value = str(selected_ts[i][var_txt])
            except:
                var_value = 'Not given'
        var_all.append(var_value)
    #
    print_sorted_list(var_all,var_txt,count_min=count_min,print_format=print_format)


# This function takes a time-lat-lon variable and computes the mean.
def global_mean(variable,lat_model,lon_model,lon_axis=2,lat_axis=1):
    #
    lat_weights = np.cos(np.radians(lat_model))
    variable_zonal = np.nanmean(variable,axis=lon_axis)
    variable_mean = np.average(variable_zonal,axis=lat_axis,weights=lat_weights)
    #
    return variable_mean


# This function takes a time-lat-lon variable and computes the mean for a given range of lon and lat.
def spatial_mean(variable,lat_model,lon_model,lat_min,lat_max,lon_min,lon_max,lon_axis=2,lat_axis=1):
    #
    j_selected = np.where((lat_model >= lat_min) & (lat_model <= lat_max))[0]
    i_selected = np.where((lon_model >= lon_min) & (lon_model <= lon_max))[0]
    print('Computing spatial mean. lats='+str(lat_model[j_selected[0]])+'-'+str(lat_model[j_selected[-1]])+', lons='+str(lon_model[i_selected[0]])+'-'+str(lon_model[i_selected[-1]])+'.  Points are inclusive.')
    #
    lat_weights = np.cos(np.radians(lat_model))
    if (lon_axis == 2) and (lat_axis == 1):
        variable_zonal = np.nanmean(variable[:,:,i_selected],axis=2)
        variable_mean = np.average(variable_zonal[:,j_selected],axis=1,weights=lat_weights[j_selected])
    elif (lon_axis == 3) and (lat_axis == 2):
        variable_zonal = np.nanmean(variable[:,:,:,i_selected],axis=3)
        variable_mean = np.average(variable_zonal[:,:,j_selected],axis=2,weights=lat_weights[j_selected])
    else:
        print('Data shape unknown. Please see function.')
        return np.nan
    #
    return variable_mean


"""
# This function takes a ens-time-lat-lon variable and computes the mean for a given range of lon and lat.
#variable,lat,lon,lat_min,lat_max,lon_min,lon_max,span_meridian = temp_seasons['jja'],lat,lon,60,90,270,30,True
def spatial_mean(variable,lat,lon,lat_min,lat_max,lon_min,lon_max,span_meridian=False):
    #
    j_selected = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    if span_meridian == False: i_selected = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    else:                      i_selected = np.where((lon >= lon_min) | (lon <= lon_max))[0]
    print('Computing spatial mean. lats='+str(lat[j_selected[0]])+'-'+str(lat[j_selected[-1]])+', lons='+str(lon[i_selected[0]])+'-'+str(lon[i_selected[-1]])+'.  Points are inclusive.')
    #
    lat_weights = np.cos(np.radians(lat))
    variable_zonal = np.nanmean(variable[:,:,i_selected],axis=2)
    variable_mean = np.average(variable_zonal[:,j_selected],axis=1,weights=lat_weights[j_selected])
    #
    return variable_mean
"""

# Compute time means from monthly data
#var,ndays_per_month = trefht_trace_monthly,ndays_per_month
def var_time_means(var,ndays_per_month):
    #
    n_months = var.shape[0]
    n_lat    = var.shape[1]
    n_lon    = var.shape[2]
    n_years  = int(n_months/12)
    #
    var_Jan = np.zeros((n_years,n_lat,n_lon)); var_Jan[:] = np.nan
    var_Jul = np.zeros((n_years,n_lat,n_lon)); var_Jul[:] = np.nan
    var_ann = np.zeros((n_years,n_lat,n_lon)); var_ann[:] = np.nan
    var_MAM = np.zeros((n_years,n_lat,n_lon)); var_MAM[:] = np.nan
    var_JJA = np.zeros((n_years,n_lat,n_lon)); var_JJA[:] = np.nan
    var_SON = np.zeros((n_years,n_lat,n_lon)); var_SON[:] = np.nan
    var_DJF_end   = np.zeros((n_years,n_lat,n_lon)); var_DJF_end[:]   = np.nan  # END OF YEAR
    var_DJF_begin = np.zeros((n_years,n_lat,n_lon)); var_DJF_begin[:] = np.nan  # BEGINNING OF YEAR
    for i in range(n_years):
        ind_jan = i*12
        #
        var_Jan[i,:,:]    = var[ind_jan,  :,:]
        var_Jul[i,:,:]    = var[ind_jan+6,:,:]
        var_ann[i,:,:]    = np.average(var[ind_jan:ind_jan+12,:,:],   axis=0,weights=ndays_per_month[ind_jan:ind_jan+12])
        var_MAM[i,:,:]    = np.average(var[ind_jan+2:ind_jan+5,:,:],  axis=0,weights=ndays_per_month[ind_jan+2:ind_jan+5])
        var_JJA[i,:,:]    = np.average(var[ind_jan+5:ind_jan+8,:,:],  axis=0,weights=ndays_per_month[ind_jan+5:ind_jan+8])
        var_SON[i,:,:]    = np.average(var[ind_jan+8:ind_jan+11,:,:], axis=0,weights=ndays_per_month[ind_jan+8:ind_jan+11])
        if i != (n_years-1):
            var_DJF_end[i,:,:]   = np.average(var[ind_jan+11:ind_jan+14,:,:],axis=0,weights=ndays_per_month[ind_jan+11:ind_jan+14])
        if i != 0:
            var_DJF_begin[i,:,:] = np.average(var[ind_jan-1:ind_jan+2,:,:],  axis=0,weights=ndays_per_month[ind_jan-1:ind_jan+2])
    #
    var_seasons = {'ann':var_ann,'mam':var_MAM,'jja':var_JJA,'son':var_SON,'djf_end':var_DJF_end,'djf_begin':var_DJF_begin,
                   'jan':var_Jan,'jul':var_Jul}
    #
    return var_seasons


# Function to average values over North America, including Greenland
#lat,lon = lat_trace,lon_trace
def mask_NorthAmerica(lat,lon,region_to_use='all'):
    #
    # Load the state boundary data
    dir_data = 'C:/Users/erbm/Dropbox/Academia/NAU/Project_EcoClimate_Sensitivity/analysis/utils/data/continents_shapefile/World_Continents_-8398826466908339531/'
    continents_shapefile = gpd.read_file(dir_data+'World_Continents.shp')
    continents_shapefile = continents_shapefile.to_crs('EPSG:4326')  # Convert to lat/lon
    #
    # Get geometry of NorthAmerica
    ind_NorthAmerica = np.where(continents_shapefile['CONTINENT'] == 'North America')[0][0]
    geom_NorthAmerica = continents_shapefile['geometry'][ind_NorthAmerica]
    #
    """
    plt.figure(figsize=(10,12))
    ax1 = plt.subplot2grid((1,1),(0,0))
    n_regions = len(geom_NorthAmerica.geoms)
    for i in range(n_regions):
        region_lons,region_lats = geom_NorthAmerica.geoms[i].exterior.coords.xy
        ax1.scatter(region_lons,region_lats)
    ax1.set_xlim(170,181); ax1.set_ylim(50,55)
    plt.show()

    # Plot the points, to explore (Note: the continental U.S. is i=476)
    n_regions = len(geom_NorthAmerica.geoms)
    for i in range(n_regions):
        region_lons,region_lats = geom_NorthAmerica.geoms[i].exterior.coords.xy
        print(min(region_lats),max(region_lats))
        print(min(region_lons),max(region_lons))
        plt.scatter(region_lons,region_lats)
        plt.title(str(i))
        plt.show()
    """
    #
    # Get the points on the grid in North America
    n_regions = len(geom_NorthAmerica.geoms)
    n_lon = len(lon)
    n_lat = len(lat)
    mask_NorthAmerica = np.zeros((n_lat,n_lon))
    for j in range(n_lat):
        print('Calculating mask: '+str(j+1)+'/'+str(n_lat))
        lat_to_check = lat[j]
        if ((lat_to_check < 5) | (lat_to_check > 85)): continue
        for i in range(n_lon): 
            lon_to_check = lon[i]
            if lon_to_check > 180: lon_to_check = lon_to_check - 360
            if ((lon_to_check > -10) & (lon_to_check < 171)): continue
            point = Point(lon_to_check,lat_to_check)
            for k in range(n_regions):
                if ((region_to_use == 'continental') & (k != 476)): continue
                geom_NorthAmerica_selected = geom_NorthAmerica.geoms[k]
                NorthAmerica_bounds = np.transpose(np.array(geom_NorthAmerica_selected.exterior.coords.xy))
                lat_min = np.min(NorthAmerica_bounds[1,:])
                lat_max = np.max(NorthAmerica_bounds[1,:])
                if ((lat_to_check < lat_min) | (lat_to_check > lat_max)): continue
                polygon_NorthAmerica = Polygon(NorthAmerica_bounds)
                if polygon_NorthAmerica.contains(point) == True: mask_NorthAmerica[j,i] = 1
    #
    return mask_NorthAmerica


#%% Compute a mean over the masked region. Input should be a time-lat-lon variable

#var,lat,lon,mask_selected = temp_20cr_seasons['ndjfm'],lat_trace,lon_trace,mask_Alaska
def mean_of_selected(var,lat,lon,mask_selected):
    #
    # Get dimensions
    n_lon = len(lon)
    n_lat = len(lat)
    #
    # Regrid to 1d
    n_time = var.shape[0]
    lon_2d,lat_2d = np.meshgrid(lon,lat)
    lat_1d           = np.reshape(lat_2d,       (n_lat*n_lon))
    mask_selected_1d = np.reshape(mask_selected,(n_lat*n_lon))
    var_1d           = np.reshape(var,   (n_time,n_lat*n_lon))
    #
    # Compute a mean of the Alaska region
    lat_weights = np.cos(np.radians(lat_1d))
    ind_to_include = np.where(mask_selected_1d == 1)[0]
    var_mean = np.average(var_1d[:,ind_to_include],axis=1,weights=lat_weights[ind_to_include])
    #
    return var_mean


#%% Get edge boundaries for lat and lon variables

#lat,lon = lat_trace,lon_trace
def get_boundaries(lat,lon):
    #
    lat_boundaries = copy.deepcopy(lat)
    lat_boundaries = np.insert(lat,0,lat[0]-(lat[1]-lat[0]))              # Add a point to the beginning
    lat_boundaries = np.append(lat_boundaries,lat[-1]+(lat[-1]-lat[-2]))  # Add a point to the end
    lat_boundaries = (lat_boundaries[:-1] + lat_boundaries[1:]) / 2       # Compute the means
    lat_boundaries[lat_boundaries >  90] =  90
    lat_boundaries[lat_boundaries < -90] = -90
    #
    lon_boundaries = copy.deepcopy(lon)
    lon_boundaries = np.insert(lon,0,lon[0]-(lon[1]-lon[0]))              # Add a point to the beginning
    lon_boundaries = np.append(lon_boundaries,lon[-1]+(lon[-1]-lon[-2]))  # Add a point to the end
    lon_boundaries = (lon_boundaries[:-1] + lon_boundaries[1:]) / 2       # Compute the means
    #
    print('Boundary ranges:')
    print('Latitude: ',lat_boundaries[0],'-',lat_boundaries[-1])
    print('Longitude:',lon_boundaries[0],'-',lon_boundaries[-1])
    #
    return lat_boundaries, lon_boundaries

#%% Select proxies based on metadata

def select_proxies(proxy_ts,interp_txt=["any"],variable_txt=["any"],archive_txt=["any"],proxy_txt=["any"],units_txt=["any"]):
    #
    selected_ts = []
    for i in range(len(proxy_ts)):
        try:    interp = proxy_ts[i]['interpretation1_variable'][0]
        except: interp = 'Not given'
        try:    variable = proxy_ts[i]['paleoData_variableName'][0]
        except: variable = 'Not given'
        try:    archive = proxy_ts[i]['archiveType'][0]
        except: archive = 'Not given'
        try:    proxy = proxy_ts[i]['paleoData_proxy'][0]
        except: proxy = 'Not given'
        try:    units = proxy_ts[i]['paleoData_units'][0]
        except: units = 'Not given'
        interp_bool  = interp   in interp_txt
        var_bool     = variable in variable_txt
        archive_bool = archive  in archive_txt
        proxy_bool   = proxy    in proxy_txt
        unit_bool    = units    in units_txt
        if interp_txt[0]   == "any": interp_bool  = True
        if variable_txt[0] == "any": var_bool     = True
        if archive_txt[0]  == "any": archive_bool = True
        if proxy_txt[0]    == "any": proxy_bool   = True
        if units_txt[0]    == "any": unit_bool    = True
        if var_bool & archive_bool & proxy_bool & interp_bool & unit_bool: selected_ts.append(proxy_ts[i])
    #
    return selected_ts
