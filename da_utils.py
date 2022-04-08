#==============================================================================
# Different function for use in the Holocene DA project.
#    author: Michael P. Erb
#    date  : 3/16/2022
#==============================================================================

import numpy as np
from scipy.linalg import sqrtm
import da_utils_lmr
import xarray as xr
import xesmf as xe

# A function to do the data assimilation.  It is based on '2_darecon.jl',
# originally written by Nathan Steiger.
#Xb,HXb,R,y = Xb,np.transpose(model_estimates_selected),R_diagonal,proxy_values_selected
def damup(Xb,HXb,R,y):
    #
    # Data assimilation matrix update step, assimilating all observations
    # for a given time step at once. Variables with their dimensions are 
    # indicated by [dim1 dim2] given below. This set of update equations
    # follow those from Whitaker and Hamill 2002: Eq. 2, 4, 10, & Sec 3.
    # ARGUMENTS:
    #     Xb = background (prior) [n_state, n_ens]
    #     HXb = model estimate of observations H(Xb) [n_proxies_valid, n_ens]
    #     y = observation (with implied noise) [n_proxies_valid]
    #     R = diagonal observation error variance matrix [n_proxies_valid, n_proxies_valid]
    #     infl = inflation factor [scalar] **Note: modify code to include**
    # RETURNS:
    #     Xa = analysis (posterior) [n_state, n_ens]
    #     Xam = analysis mean [n_state]
    #
    # Number of ensemble members
    nens = Xb.shape[1] 
    #
    # Decompose Xb and HXb into mean and perturbations (for Eqs. 4 & Sec 3)
    Xbm = np.mean(Xb,axis=1)
    Xbp = Xb - Xbm[:,None]
    #
    HXbm = np.mean(HXb,axis=1)
    HXbp = HXb - HXbm[:,None]
    #
    # Kalman gain for mean and matrix covariances (Eq. 2)
    PbHT   = np.dot(Xbp, np.transpose(HXbp))/(nens-1)
    HPbHTR = np.dot(HXbp,np.transpose(HXbp))/(nens-1)+R
    K = np.dot(PbHT,np.linalg.inv(HPbHTR))
    #
    # Kalman gain for the perturbations (Eq. 10)
    sHPbHTR = sqrtm(HPbHTR)
    sR      = sqrtm(R)
    Ktn = np.dot(PbHT,np.transpose(np.linalg.inv(sHPbHTR)))
    Ktd = np.linalg.inv(sHPbHTR+sR)
    Kt = np.dot(Ktn,Ktd)
    #
    # Update mean and perturbations (Eq. 4 & Sec 3)
    Xam = Xbm + np.dot(K,(y-HXbm))
    Xap = Xbp - np.dot(Kt,HXbp)
    #
    # Reconstitute the full ensemble state vector
    Xa = Xap + Xam[:,None]
    #
    # Output both the full ensemble and the ensemble mean
    return Xa,Xam,K


# Read in a string and a lat and return a set of months
def interpret_seasonality(seasonality_txt,lat,unknown_option):
    #
    # Terms which are not dependent on latitude
    if   (str(seasonality_txt).lower() == 'annual'):                        seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'ann'):                           seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'annua'):                         seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'tann; 2'):                       seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'year'):                          seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == '1 2 3 4 5 6 7 8 9 10 11 12'):    seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'subannual'):                     seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'nan'):                           seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'not specified'):                 seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'n/a'):                           seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'n/a (subannually resolved)'):    seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'not indicated'):                 seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'aug+ann'):                       seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == '2 2 3 4 5 6 7 8 9 10 11 12'):    seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'annual (but 80% of precipitation from nov to may)'): seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == 'jan'):                           seasonality = '1'
    elif (str(seasonality_txt).lower() == 'may'):                           seasonality = '5'
    elif (str(seasonality_txt).lower() == 'july'):                          seasonality = '7'
    elif (str(seasonality_txt).lower() == 'july air'):                      seasonality = '7'
    elif (str(seasonality_txt).lower() == 'jul'):                           seasonality = '7'
    elif (str(seasonality_txt).lower() == 'tjul'):                          seasonality = '7'
    elif (str(seasonality_txt).lower() == 'mean july air temperature estimate'): seasonality = '7'
    elif (str(seasonality_txt).lower() == '7'):                             seasonality = '7'
    elif (str(seasonality_txt).lower() == '8'):                             seasonality = '8'
    elif (str(seasonality_txt).lower() == 'jul+jan'):                       seasonality = '1 7'
    elif (str(seasonality_txt).lower() == 'warmest+coldest'):               seasonality = '1 7'
    elif (str(seasonality_txt).lower() == 'coldest+warmest'):               seasonality = '1 7'
    elif (str(seasonality_txt).lower() == 'warmest + coldest'):             seasonality = '1 7'
    elif (str(seasonality_txt).lower() == 'warmest + coldest months'):      seasonality = '1 7'
    elif (str(seasonality_txt).lower() == 'warmest + coldest month'):       seasonality = '1 7'
    elif (str(seasonality_txt).lower() == 'coldest + warmest month'):       seasonality = '1 7'
    elif (str(seasonality_txt).lower() == 'jja'):                           seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'jjas'):                          seasonality = '6 7 8 9'
    elif (str(seasonality_txt).lower() == '6 7 8 9 10'):                    seasonality = '6 7 8 9 10'
    elif (str(seasonality_txt).lower() == '6 7 8 9'):                       seasonality = '6 7 8 9'
    elif (str(seasonality_txt).lower() == '6 7 8'):                         seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == '5'):                             seasonality = '5'
    elif (str(seasonality_txt).lower() == 'aug'):                           seasonality = '8'
    elif (str(seasonality_txt).lower() == 'summer (may-oct)'):              seasonality = '5 6 7 8 9 10'
    elif (str(seasonality_txt).lower() == 'summer (6,7,8)'):                seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'winter+summer'):                 seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'winter + summer'):               seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'winter; summer'):                seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'summer + winter'):               seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'summer and winter'):             seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'dec-feb'):                       seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == '6 7 2008'):                      seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == '7 8 2009'):                      seasonality = '7 8 9'
    elif (str(seasonality_txt).lower() == '12 1 2002'):                     seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == '1'):                             seasonality = '1'
    elif (str(seasonality_txt).lower() == '1 (summer)'):                    seasonality = '1'
    elif (str(seasonality_txt).lower() == '1; summer'):                     seasonality = '1'
    elif (str(seasonality_txt).lower() == '2'):                             seasonality = '2'
    elif (str(seasonality_txt).lower() == '1; 7'):                          seasonality = '1 7'
    elif (str(seasonality_txt).lower() == '1 7'):                           seasonality = '1 7'
    elif (str(seasonality_txt).lower() == '1,7'):                           seasonality = '1 7'
    elif (str(seasonality_txt).lower() == '1,8'):                           seasonality = '1 8'
    elif (str(seasonality_txt).lower() == '1,9'):                           seasonality = '1 9'
    elif (str(seasonality_txt).lower() == '1,10'):                          seasonality = '1 10'
    elif (str(seasonality_txt).lower() == '1,11'):                          seasonality = '1 11'
    elif (str(seasonality_txt).lower() == '4 5 6 7 8 9 10 11 12'):          seasonality = '4 5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == '4 5 6 7 8 9 10 12'):             seasonality = '4 5 6 7 8 9 10 12'
    elif (str(seasonality_txt).lower() == '5 6 7 8 9 10 11 12'):            seasonality = '5 6 7 8 9 10 11 12'
    elif (str(seasonality_txt).lower() == '12 1 2 3 4 5'):                  seasonality = '-12 1 2 3 4 5'
    elif (str(seasonality_txt).lower() == '11 12 1 2 3 4 5'):               seasonality = '-11 -12 1 2 3 4 5'
    elif (str(seasonality_txt).lower() == '-11 -12 1 2 3 4 5'):             seasonality = '-11 -12 1 2 3 4 5'
    elif (str(seasonality_txt).lower() == '12 1 2; 6 7 8'):                 seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == '1 2 3'):                         seasonality = '1 2 3'
    elif (str(seasonality_txt).lower() == '6 7'):                           seasonality = '6 7'
    elif (str(seasonality_txt).lower() == '7 8 9'):                         seasonality = '7 8 9'
    elif (str(seasonality_txt).lower() == '8 9 10'):                        seasonality = '8 9 10'
    elif (str(seasonality_txt).lower() == '12 1 2; 6 7 8'):                 seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == '12 1 2'):                        seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == '6 7 8 9 10 11'):                 seasonality = '6 7 8 9 10 11'
    elif (str(seasonality_txt).lower() == '9 10 11 12 1 2 3 4 5 6 7'):      seasonality = '-9 -10 -11 -12 1 2 3 4 5 6 7'
    elif (str(seasonality_txt).lower() == '-9 -10 -11 -12 1 2 3 4 5 6 7'):  seasonality = '-9 -10 -11 -12 1 2 3 4 5 6 7'
    elif (str(seasonality_txt).lower() == '5 6 7 8 9 10'):                  seasonality = '5 6 7 8 9 10'
    elif (str(seasonality_txt).lower() == '5 6 7 8 9 10 11'):               seasonality = '5 6 7 8 9 10 11'
    elif (str(seasonality_txt).lower() == '6 7 8 9 10 11'):                 seasonality = '6 7 8 9 10 11'
    elif (str(seasonality_txt).lower() == '-12 1 2 6 7 8'):                 seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == '-12 1 2'):                       seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == '-12 1 2 3 4 5'):                 seasonality = '-12 1 2 3 4 5'
    elif (str(seasonality_txt).lower() == '1 6 7 8'):                       seasonality = '1 6 7 8'
    elif (str(seasonality_txt).lower() == '-12 1 2 7'):                     seasonality = '-12 1 2 7'
    elif (str(seasonality_txt).lower() == '12,1,2'):                        seasonality = '12 1 2'
    elif (str(seasonality_txt).lower() == '1,2,3'):                         seasonality = '1 2 3'
    elif (str(seasonality_txt).lower() == '6,7'):                           seasonality = '6 7'
    elif (str(seasonality_txt).lower() == '6,7,8'):                         seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == '7,8,9'):                         seasonality = '7 8 9'
    elif (str(seasonality_txt).lower() == '8,9,10'):                        seasonality = '8 9 10'
    elif (str(seasonality_txt).lower() == '((( 6 7 2008 ))) 6 7 8 /// 6 7 8'):    seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == '((( 7 8 2009 ))) 7 8 9 /// 7 8 9'):    seasonality = '7 8 9'
    elif (str(seasonality_txt).lower() == '((( 12 1 2002 ))) 12 1 2 /// 12 1 2'): seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == '((( 1 2 2003 ))) 1 2 3 /// 1 2 3'):    seasonality = '1 2 3'
    #
    # Terms which are dependent on latitude
    elif (str(seasonality_txt).lower() == 'summer')                  and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'summer')                  and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'mean summer')             and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'mean summer')             and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'warmest quarter yr')      and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'warmest quarter yr')      and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'growing')                 and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'growing')                 and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'growing season')          and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'growing season')          and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'warm season')             and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'warm season')             and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'winter')                  and (lat >= 0): seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'winter')                  and (lat < 0):  seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'cold season')             and (lat >= 0): seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'cold season')             and (lat < 0):  seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'autumn')                  and (lat >= 0): seasonality = '9 10 11'
    elif (str(seasonality_txt).lower() == 'autumn')                  and (lat < 0):  seasonality = '3 4 5'
    elif (str(seasonality_txt).lower() == 'warmest month + winter')  and (lat >= 0): seasonality = '-12 1 2 7'
    elif (str(seasonality_txt).lower() == 'warmest month + winter')  and (lat < 0):  seasonality = '1 6 7 8'
    elif (str(seasonality_txt).lower() == 'coldest month + summer')  and (lat >= 0): seasonality = '1 6 7 8'
    elif (str(seasonality_txt).lower() == 'coldest month + summer')  and (lat < 0):  seasonality = '-12 1 2 7'
    elif (str(seasonality_txt).lower() == 'mtwa')                    and (lat >= 0): seasonality = '7'  # MTWA: Mean Temperature of the Warmest Month
    elif (str(seasonality_txt).lower() == 'mtwa')                    and (lat < 0):  seasonality = '1'  # MTWA: Mean Temperature of the Warmest Month
    elif (str(seasonality_txt).lower() == 'warmest')                 and (lat >= 0): seasonality = '7'
    elif (str(seasonality_txt).lower() == 'warmest')                 and (lat < 0):  seasonality = '1'
    elif (str(seasonality_txt).lower() == 'warmest month')           and (lat >= 0): seasonality = '7'
    elif (str(seasonality_txt).lower() == 'warmest month')           and (lat < 0):  seasonality = '1'
    elif (str(seasonality_txt).lower() == 'coldest')                 and (lat >= 0): seasonality = '1'
    elif (str(seasonality_txt).lower() == 'coldest')                 and (lat < 0):  seasonality = '7'
    elif (str(seasonality_txt).lower() == 'coldest month')           and (lat >= 0): seasonality = '1'
    elif (str(seasonality_txt).lower() == 'coldest month')           and (lat < 0):  seasonality = '7'
    elif (str(seasonality_txt).lower() == 'early summer')            and (lat >= 0): seasonality = '6 7'
    elif (str(seasonality_txt).lower() == 'early summer')            and (lat < 0):  seasonality = '-12 1'
    elif (str(seasonality_txt).lower() == 'winter/spring')           and (lat >= 0): seasonality = '-12 1 2 3 4 5'
    elif (str(seasonality_txt).lower() == 'winter/spring')           and (lat < 0):  seasonality = '6 7 8 9 10 11'
    elif (str(seasonality_txt).lower() == 'summer and early autumn') and (lat >= 0): seasonality = '6 7 8 9 10'
    elif (str(seasonality_txt).lower() == 'summer and early autumn') and (lat < 0):  seasonality = '-12 1 2 3 4'
    elif (str(seasonality_txt).lower() == 'summer + early autumn')   and (lat >= 0): seasonality = '6 7 8 9 10'
    elif (str(seasonality_txt).lower() == 'summer + early autumn')   and (lat < 0):  seasonality = '-12 1 2 3 4'
    elif (str(seasonality_txt).lower() == 'mean temperature of the warmest quarter (twarm)') and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'mean temperature of the warmest quarter (twarm)') and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'spring-fall')             and (lat >= 0): seasonality = '3 4 5 6 7 8 9 10 11'
    elif (str(seasonality_txt).lower() == 'spring-fall')             and (lat < 0):  seasonality = '-9 -10 -11 -12 1 2 3 4 5'
    #
    # Unsure if this term is dependent on latitude
    elif (str(seasonality_txt).lower() == 'spring/fall') and (lat >= 0): seasonality = '3 4 5 9 10 11'
    #
    # Try to parse the seasonality as numbers. If that doesn't work, use the default value or return as-is.
    else:
        text_list = str(seasonality_txt).split(' ')
        if all(isinstance(i,int) for i in text_list):
            print('ATTENTION! Seasonality text unknown, parsing as numbers. Seasonality: |'+str(seasonality_txt)+'|')
            seasonality = np.array([int(i) for i in text_list])
        elif unknown_option == 'annual':
            print('ATTENTION! Seasonality text unknown, using annual.       Seasonality: |'+str(seasonality_txt)+'|')
            seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
        else:
            print('ATTENTION! Seasonality text unknown, returning as-is.    Seasonality: |'+str(seasonality_txt)+'|')
            seasonality = seasonality_txt
    #
    return seasonality


# A function to regrid an age-month-lat-lon array to a standardized grid
def regrid_model(var,lat,lon,age,regrid_method='conservative_normed',make_figures=False):
    #
    # Put the data in an xarray
    var_xarray = xr.Dataset(
        {
            'variable':(['age','month','lat','lon'],var)
        },
        coords={
            'lat':   (['lat'],lat,{'units':'degrees_north'}),
            'lon':   (['lon'],lon,{'units':'degrees_east'}),
            'month': (['month'],np.arange(1,13)),
            'age':   (['age'],age),
        },
    )
    #
    # Set up output variable on a 96 x 64 grid
    lat_regrid = np.arange(-88.59375,90,2.8125)
    lon_regrid = np.arange(0,360,3.75)
    data_format = xr.Dataset(
        {
            'lat': (['lat'],lat_regrid,{'units':'degrees_north'}),
            'lon': (['lon'],lon_regrid,{'units':'degrees_east'}),
        }
    )
    #
    # Set up the regridder and do the regridding
    regridder = xe.Regridder(var_xarray,data_format,regrid_method,periodic=True)
    var_regridded = regridder(var_xarray,keep_attrs=True)
    #
    """
    # Figures to compare the original data to the regridded data
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    #
    ax1 = plt.subplot2grid((2,1),(0,0),projection=ccrs.PlateCarree())
    ax2 = plt.subplot2grid((2,1),(1,0),projection=ccrs.PlateCarree())
    var_xarray.variable.isel(age=0,month=0).plot.pcolormesh(ax=ax1)
    var_regridded.variable.isel(age=0,month=0).plot.pcolormesh(ax=ax2)
    ax1.coastlines()
    ax2.coastlines()
    plt.plot()
    #
    ax1 = plt.subplot2grid((1,1),(0,0))
    var_xarray.variable.isel(age=0,month=0,lon=0).plot(ax=ax1)
    var_regridded.variable.isel(age=0,month=0,lon=0).plot(ax=ax1)
    plt.plot()
    """
    #
    # Get the regridded values
    var_regrid = var_regridded.variable.values
    lat_new    = var_regridded.lat.values
    lon_new    = var_regridded.lon.values
    #
    return var_regrid,lat_new,lon_new


# This function takes an array and computes the global-mean, after being told the axes of lat and lon.
def global_mean(variable,lats,index_lat,index_lon):
    variable_zonal = np.nanmean(variable,axis=index_lon)
    if index_lon < index_lat: index_lat = index_lat-1
    lat_weights = np.cos(np.radians(lats))
    variable_global = np.average(variable_zonal,axis=index_lat,weights=lat_weights)
    return variable_global


# This function takes a time-lat-lon variable and computes the mean for a given range of lon and lat.
def spatial_mean(variable,lats,lons,lat_min,lat_max,lon_min,lon_max,index_lat,index_lon,verbose=False):
    #
    j_selected = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    i_selected = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    if verbose: print('Computing spatial mean. lats='+str(lats[j_selected[0]])+'-'+str(lats[j_selected[-1]])+', lons='+str(lons[i_selected[0]])+'-'+str(lons[i_selected[-1]])+'.  Points are inclusive.')
    #
    if   index_lon == 1: variable_zonal = np.nanmean(variable[:,i_selected],    axis=1)
    elif index_lon == 2: variable_zonal = np.nanmean(variable[:,:,i_selected],  axis=2)
    elif index_lon == 3: variable_zonal = np.nanmean(variable[:,:,:,i_selected],axis=3)
    else: print('Invalid lon dimension chosen'); return None
    #
    lat_weights = np.cos(np.radians(lats))
    if index_lon < index_lat: index_lat = index_lat-1
    if   index_lat == 0: variable_mean = np.average(variable_zonal[j_selected],    axis=0,weights=lat_weights[j_selected])
    elif index_lat == 1: variable_mean = np.average(variable_zonal[:,j_selected],  axis=1,weights=lat_weights[j_selected])
    elif index_lat == 2: variable_mean = np.average(variable_zonal[:,:,j_selected],axis=2,weights=lat_weights[j_selected])
    else: print('Invalid lat dimension chosen'); return None
    #
    return variable_mean


# A function to compute a localization matrix
def loc_matrix(options,model_data,proxy_data):
    #
    lat_model = model_data['lat']
    lon_model = model_data['lon']
    #
    # Get dimensions
    n_proxies = proxy_data['values_binned'].shape[0]
    n_vars    = len(options['vars_to_reconstruct'])
    n_latlon  = len(lat_model) * len(lon_model)
    n_state = (n_latlon*n_vars) + n_proxies
    #
    # Compute the localization values for every proxy
    proxy_localization_all = np.ones((n_proxies,n_state))
    if options['localization_radius'] != 'None':
        #
        # Get lat and lon values for the prior
        lon_model_2d,lat_model_2d = np.meshgrid(lon_model,lat_model)
        lat_prior = np.reshape(lat_model_2d,(n_latlon))
        lon_prior = np.reshape(lon_model_2d,(n_latlon))
        prior_coords = np.concatenate((lat_prior[:,None],lon_prior[:,None]),axis=1)
        #
        # Repeat the prior coords for all reconstructed variables
        if n_vars > 1: prior_coords = np.tile(prior_coords,(n_vars,1))
        #
        # Include the proxy coordinates with the model coordinates
        proxy_coords_all = np.zeros((n_proxies,2)); proxy_coords_all[:] = np.nan
        for i in range(n_proxies):
            proxy_coords_all[i,0] = proxy_data['lats'][i]
            proxy_coords_all[i,1] = proxy_data['lons'][i]
        #
        prior_coords = np.append(prior_coords,proxy_coords_all,axis=0)
        #
        for i in range(n_proxies):
            #
            # Get proxy metdata
            proxy_lat = proxy_data['lats'][i]
            proxy_lon = proxy_data['lons'][i]
            #
            # Compute the localization values and save it to a common variable
            #locRad, proxy_lat, proxy_lon, X_coords = options['localization_radius'],proxy_lat,proxy_lon,prior_coords
            proxy_localization = da_utils_lmr.cov_localization(options['localization_radius'],proxy_lat,proxy_lon,prior_coords)
            proxy_localization_all[i,:] = proxy_localization
    #
    return proxy_localization_all

