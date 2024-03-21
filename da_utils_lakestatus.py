import scipy
import xarray as xr
import numpy  as np
import pyet
import metpy.calc as mpcalc

#A function to calculate skill between two xarray dataarrays 
def calcSkill(array1,array2,method='r',dim='time',calcMean=True,w=False):
    #Calcualte
    if method == 'r':
        skill = xr.corr(array1,array2,dim=dim)
    elif method == 'r2':
        skill = xr.corr(array1,array2,dim=dim)**2
    elif method == 'RMSE':
        squared_diff = (array1 - array2)**2
        mse = squared_diff.mean(dim=dim)
        skill = xr.ufuncs.sqrt(mse)
    #Calculate Weighted mean
    if calcMean: 
        if w: skill = np.round(skill.weighted(np.cos(np.deg2rad(array1.lat))).mean().data,2)
        else: skill = np.round(skill.mean().data,2)
    #
    return(skill)

#A function to convert xarray dataarray to percentile units
def lake2percentile(inarray):
    if 'season' in inarray.dims:
        for szni in range(len(inarray.season)):
            for lati in range(len(inarray.lat)):
                for loni in range(len(inarray.lon)): 
                    if np.sum(np.isfinite(inarray[:,szni,lati,loni]))>0:
                        ranks = scipy.stats.mstats.rankdata(np.ma.masked_invalid(inarray[:,szni,lati,loni]))                                        #Additional step to mask nan values
                        ranks[ranks == 0] = np.nan
                        if np.sum(np.isfinite(ranks)) > 1:
                            ranks-=1
                            ranks/= np.sum(np.isfinite(ranks))-1
                        else: ranks*=np.NaN
                        inarray[:,szni,lati,loni]=ranks
    else: 
        for lati in range(len(inarray.lat)):
            for loni in range(len(inarray.lon)):
                if np.sum(np.isfinite(inarray[:, lati, loni])) > 0:
                     ranks = scipy.stats.mstats.rankdata(np.ma.masked_invalid(inarray[:, lati, loni]))  # Additional step to mask nan values
                     ranks[ranks == 0] = np.nan
                     if np.sum(np.isfinite(ranks)) > 1:
                         ranks -= 1
                         ranks /= np.sum(np.isfinite(ranks))-1
                     else: ranks *= np.NaN
                     inarray[:, lati, loni] = ranks
    return(inarray)

#A function to calculate lake status values in percentile units
def calcLakeStatus(runoff=None,precip=None,evap=None,levap=None,Qmethod='P/E',maskArray=False,percentile=True):
    # Calculate runoff
    if   Qmethod == 'P/E': runoff = precip/evap
    elif Qmethod == 'P-E': runoff = precip-evap
    elif Qmethod == 'P': runoff = precip
    elif Qmethod == 'P^2/E': runoff = (precip**2)/evap
    elif Qmethod == 'runoff': runoff = runoff
    else: runoff = None
    runoff = runoff.where(runoff>0,runoff,0)
    # Calculate lakestatus as ratio
    lakeStatus = runoff/(levap-precip)
    # land or snow mask
    #if len(np.shape(False)) > 0: lakeStatus = lakeStatus.where(maskArray==True)
    # Convert to percentile (0-1)
    if percentile:
        positive = lake2percentile(lakeStatus.where(lakeStatus>=0))
        negative = lake2percentile(lakeStatus.where(lakeStatus<0))
        negative+= 1.1
        vals = np.where(np.isfinite(positive),positive,negative)
        lakeStatus.data = vals 
        lakeStatus =lake2percentile(lakeStatus)
    return(lakeStatus)
    #

# A function to reorganize data provided by carrie morrill ()to compare with raw CCSM4 data
def lm19_2_gcmFromat(array): 
    #Bin annually
    time_bins = list(array.time[range(0,len(array.time),12)].data)
    time_bins.append(float(max(array.time))+1)
    array = array.groupby_bins('time', bins=time_bins, right=False).mean(dim='time').rename({'time_bins':'time'})
    array['time'] = range(0,100)
    array=array.where(array != -99.90002).transpose('time','lat','lon')
    return(array)

#Create a function which uses pyet and mpcalc to calculate PET using priestley_taylor or penman methods and xarray dataarrays
def calculatePET(method='priestley_taylor', tas = None, press = None, relhum=None, spehum=None,
                 netrad=None, netSW=None, netLW=None, downSW=None, toaSW=None, elev=None,
                 wind=None ):
    # Potential evaporation mm/day.

    # Parameters
    # ----------
    # method string, required (one of 'priestley_taylor' or 'penman')
    #     equation to use
    # tas: xarray.DataArray, required
    #     average day temperature [°C]
    # press: xarray.DataArray, required
    #     surface pressure [kPa]
    # relhum: xarray.DataArray, optional (need one of relhum or spehum (preferably relhum))
    #     relative humidity [%]
    # spehum: xarray.DataArray, optional (need one of relhum or spehum (preferably relhum))
    #     surface humidity [kg/kg]
    # netrad: xarray.DataArray, optional [MJ/(m2*day)]
    #     net radiation
    # spehum: xarray.DataArray, optional (need one of relhum or spehum (preferably relhum))
    #     wind [,/s]

    # Notes
    # -----
    # Modified from pyet to not need datatime age configuration
    # Based on equation 39 in :cite:t:`allen_crop_1998`.
    #
    #Calculate humidity
    if relhum is None:
        if spehum is None:
            print('!!!ERROR: One of relhum or spehum must be provided!!!!')
            #return None
        else:
            print('estimating relative humidity from specific humidity using mpcalc.relative_humidity_from_specific_humidity')
            rh = mpcalc.relative_humidity_from_specific_humidity(
                        pressure          = press,
                        temperature       = tas,
                        specific_humidity = spehum)#.to('percent')
            rh = rh.clip(0,1) *100
    else: rh = relhum
    #
    #Calcualte Net Radiation
    if netrad is None:
        #Need shortwave
        print('calculating net radiation from available radiation paramaters provided')
        if netSW is None: 
            if downSW is None: 
                print('!!!ERROR: One of netrad, netSW, or downSW!!!')
                return None
            else: SWn = downSW*0.9 #estimate albedo of water
        else: SWn = netSW
        #and longwave
        if netLW is None:  #Could cause error if don't provide correct data
            #modified from pyet.calc_rad_long() :cite:t:`allen_crop_1998`.
            a=1.35; b=-0.35
            STEFAN_BOLTZMANN_DAY = 4.903 * 10 ** -9 
            ea = pyet.calc_ea(tmean=tas,rh=rh)  
            ea.data = np.array(ea.data)
            rso =  (0.75 + (2 * 10 ** -5) * elev) * toaSW #modified of calc_rso
            solar_rat = np.clip(downSW / rso, 0.3, 1)
            tmp1 = STEFAN_BOLTZMANN_DAY * (tas + 273.16) ** 4
            tmp2 = 0.34 - 0.14 * np.sqrt(ea)  # OK
            tmp3 = a * solar_rat + b  # OK
            tmp3 = np.clip(tmp3, 0.05, 1) 
            LWn = (tmp1 * tmp2 * tmp3)
        else: LWn = netLW
        rn = SWn - LWn        
    else: rn = netrad
    #
    #Calculate PET
    if method == 'priestley_taylor':
        pet = pyet.priestley_taylor(tmean=tas,rn=rn,rh=rh,pressure=press)
        pet.attrs['long_name'] =  'Potential evapotranspiration (priestley_assessment_1972)' 
    elif method == 'penman':
        pet = pyet.penman(tmean=tas,rn=rn,rh=rh,pressure=press,wind=wind)
        pet.attrs['long_name'] =  'Potential evapotranspiration (penman_natural_1948)' 
    else: 
        print('!!!ERROR: Chosen method is not available. Choose one of priestley_taylor or penman!!!!')
        return None
    #Return PET dataarray
    pet.attrs['units'] = "mm/day"
    try: 
        pet=pet.reset_coords(['toa','ht'], drop=True).rename('PET')
    except:
        pet = pet.rename('PET')
    return pet
  #                                        