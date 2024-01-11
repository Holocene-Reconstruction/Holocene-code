import scipy
import xarray as xr
import numpy  as np

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

