#==============================================================================
# Different function for use in the Holocene DA project.
#    author: Michael P. Erb
#    date  : 9/16/2020
#==============================================================================

import numpy as np
from scipy.linalg import sqrtm
import da_utils_lmr

# A function to do the data assimilation.  It is based on '2_darecon.jl',
# originally written by Nathan Steiger.
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
    # The climate interpretation should be represented as a span of months.
    #
    # Not dependant on latitude
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
    elif (str(seasonality_txt).lower() == 'warmest+coldest'):               seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'coldest+warmest'):               seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'warmest + coldest'):             seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'warmest + coldest months'):      seasonality = '-12 1 2 6 7 8'
    elif (str(seasonality_txt).lower() == 'dec-feb'):                       seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == '6 7 2008'):                      seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == '7 8 2009'):                      seasonality = '7 8 9'
    elif (str(seasonality_txt).lower() == '12 1 2002'):                     seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == '1 (summer)'):                    seasonality = '1'
    elif (str(seasonality_txt).lower() == '1; summer'):                     seasonality = '1'
    elif (str(seasonality_txt).lower() == '1'):                             seasonality = '1'
    elif (str(seasonality_txt).lower() == '2'):                             seasonality = '2'
    elif (str(seasonality_txt).lower() == '1; 7'):                          seasonality = '1 7'
    elif (str(seasonality_txt).lower() == '1 7'):                           seasonality = '1 7'
    elif (str(seasonality_txt).lower() == '1,7'):                           seasonality = '1 7'
    elif (str(seasonality_txt).lower() == '1,8'):                           seasonality = '1 8'
    elif (str(seasonality_txt).lower() == '1,9'):                           seasonality = '1 9'
    elif (str(seasonality_txt).lower() == '1,10'):                          seasonality = '1 10'
    elif (str(seasonality_txt).lower() == '1,11'):                          seasonality = '1 11'
    elif (str(seasonality_txt).lower() == 'warmest + coldest month'):       seasonality = '1 7'
    elif (str(seasonality_txt).lower() == 'coldest + warmest month'):       seasonality = '1 7'
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
    #elif (str(seasonality_txt).lower() == '2 3 4 5 6 7 8'):                       seasonality = '2 3 4 5 6 7 8'
    #elif (str(seasonality_txt).lower() == '-5 -6 -7 -8 -9 -10 -11 -12 1 2 3 4'):  seasonality = '-5 -6 -7 -8 -9 -10 -11 -12 1 2 3 4'
    #elif (str(seasonality_txt).lower() == '-12 -11 -10 -9 -8 1 2 3 4 5 6 7'):     seasonality = '-12 -11 -10 -9 -8 1 2 3 4 5 6 7'
    #elif (str(seasonality_txt).lower() == '4 5 6 7 8 9'):                         seasonality = '4 5 6 7 8 9'
    #
    # Dependant on latitude
    elif (str(seasonality_txt).lower() == 'summer')                  and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'summer')                  and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'mean summer')             and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'mean summer')             and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'warmest quarter yr')      and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'warmest quarter yr')      and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'winter')                  and (lat >= 0): seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'winter')                  and (lat < 0):  seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'autumn')                  and (lat >= 0): seasonality = '9 10 11'
    elif (str(seasonality_txt).lower() == 'autumn')                  and (lat < 0):  seasonality = '3 4 5'
    elif (str(seasonality_txt).lower() == 'growing')                 and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'growing')                 and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'warm season')             and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'warm season')             and (lat < 0):  seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'cold season')             and (lat >= 0): seasonality = '-12 1 2'
    elif (str(seasonality_txt).lower() == 'cold season')             and (lat < 0):  seasonality = '6 7 8'
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
    elif (str(seasonality_txt).lower() == 'growing season')          and (lat >= 0): seasonality = '6 7 8'
    elif (str(seasonality_txt).lower() == 'growing season')          and (lat < 0):  seasonality = '-12 1 2'
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
    # Unsure if dependant on latitude
    elif (str(seasonality_txt).lower() == 'spring/fall') and (lat >= 0): seasonality = '3 4 5 9 10 11'
    #
    else:
        if unknown_option == 'annual':
            print('ATTENTION!  Seasonality text unknown, using annual.  Seasonality: |'+str(seasonality_txt)+'|')
            seasonality = '1 2 3 4 5 6 7 8 9 10 11 12'
        else:
            print('ATTENTION!  Seasonality text unknown, returning as-is.  Seasonality: |'+str(seasonality_txt)+'|')
            seasonality = seasonality_txt
    #
    return seasonality

"""
# Add something like the following to the end of the function above:
    try:
        text = seasonality.split(' ')
        climateInterp = np.array([int(i) for i in text])
    except:
        print('ATTENTION!  Cannot parse seasonality metadata for proxy '+str(index)+': '+proxyID+'.  Seasonality: '+str(seasonality_txt))
        climateInterp = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
"""

# A function to regrid an age-month-lat-lon array to a standardized grid
def regrid_model(var,lat,lon,age,regrid_value=64):
    #
    # Old grid
    # Repeat the westernmost points on the easternmost side, to account for looping.
    var = np.append(var,var[:,:,:,0,None],axis=3)
    lon = np.append(lon,lon[0,None]+360,axis=0)
    lon_2d,lat_2d = np.meshgrid(lon,lat)
    #
    # For regridding, the array must be 2D with the shape [nlat*nlon,nens].
    # Here, nens with be both age and month.
    ntime  = var.shape[0]
    nmonth = var.shape[1]
    nlat   = var.shape[2]
    nlon   = var.shape[3]
    var_2d = np.reshape(var,(ntime*nmonth,nlat*nlon))
    var_2d = np.rollaxis(var_2d,1,0)
    #
    # Regrid the data
    lats_lons = np.column_stack((lat_2d.flatten(),lon_2d.flatten()))
    [var_regrid_2d,lat_new_2d,lon_new_2d] = da_utils_lmr.regrid_simple(ntime*nmonth,var_2d,lats_lons,0,1,regrid_value)
    #
    # Reshape the data back to a lat-lon grid
    lat_new = np.mean(lat_new_2d,axis=1)
    lon_new = np.mean(lon_new_2d,axis=0)
    nlat_new = len(lat_new)
    nlon_new = len(lon_new)
    var_regrid_2d = np.rollaxis(var_regrid_2d,1,0)
    var_regrid = np.reshape(var_regrid_2d,(ntime,nmonth,nlat_new,nlon_new))
    #
    """
    # Checking the results
    var_regrid_annual = np.mean(np.mean(var_regrid,axis=0),axis=0)
    plt.contourf(lon_new,lat_new,var_regrid_annual)
    plt.plot(var_regrid[-1,:,45,67])
    """
    #
    return var_regrid,lat_new,lon_new

# This function takes an array and computes the global-mean, after being told the axes of lat and lon.
def global_mean(variable,lats,index_lat,index_lon):
    variable_zonal = np.nanmean(variable,axis=index_lon)
    if index_lon < index_lat: index_lat = index_lat-1
    lat_weights = np.cos(np.radians(lats))
    variable_global = np.average(variable_zonal,axis=index_lat,weights=lat_weights)
    return variable_global

