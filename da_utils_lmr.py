#==========================================================================================
# This script contains functions developed for the LMR project.
#
# Source files in the LMR project for each function:
# LMR_DA.py 
#    - enkf_update_array (unaltered)
#    - cov_localization  (minor changes)
# LMR_lite_utils.py
#    - Kalman_optimal (unaltered, todo note added)
# LMR_utils.py 
#    - haversine:            (unaltered)
#    - generate_latlon       (unaltered)
#    - calculate_latlon_bnds (unaltered)
#    - lon_lat_to_cartesian  (unaltered)
#    - regrid_simple         (unaltered)
#
#==========================================================================================

import numpy as np
from time import time
from scipy.spatial import cKDTree

def enkf_update_array(Xb, obvalue, Ye, ob_err, loc=None, inflate=None):
    """
    Function to do the ensemble square-root filter (EnSRF) update
    (ref: Whitaker and Hamill, Mon. Wea. Rev., 2002)

    Originator: G. J. Hakim, with code borrowed from L. Madaus
                Dept. Atmos. Sciences, Univ. of Washington

    Revisions:

    1 September 2017: 
                    - changed varye = np.var(Ye) to varye = np.var(Ye,ddof=1) 
                    for an unbiased calculation of the variance. 
                    (G. Hakim - U. Washington)
    
    -----------------------------------------------------------------
     Inputs:
          Xb: background ensemble estimates of state (Nx x Nens) 
     obvalue: proxy value
          Ye: background ensemble estimate of the proxy (Nens x 1)
      ob_err: proxy error variance
         loc: localization vector (Nx x 1) [optional]
     inflate: scalar inflation factor [optional]
    """

    # Get ensemble size from passed array: Xb has dims [state vect.,ens. members]
    Nens = Xb.shape[1]

    # ensemble mean background and perturbations
    xbm = np.mean(Xb,axis=1)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    # ensemble mean and variance of the background estimate of the proxy 
    mye   = np.mean(Ye)
    varye = np.var(Ye,ddof=1)

    # lowercase ye has ensemble-mean removed 
    ye = np.subtract(Ye, mye)

    # innovation
    try:
        innov = obvalue - mye
    except:
        print('innovation error. obvalue = ' + str(obvalue) + ' mye = ' + str(mye))
        print('returning Xb unchanged...')
        return Xb
    
    # innovation variance (denominator of serial Kalman gain)
    kdenom = (varye + ob_err)

    # numerator of serial Kalman gain (cov(x,Hx))
    kcov = np.dot(Xbp,np.transpose(ye)) / (Nens-1)

    # Option to inflate the covariances by a certain factor
    #if inflate is not None:
    #    kcov = inflate * kcov # This implementation is not correct. To be revised later.

    # Option to localize the gain
    if loc is not None:
        kcov = np.multiply(kcov,loc) 
   
    # Kalman gain
    kmat = np.divide(kcov, kdenom)

    # update ensemble mean
    xam = xbm + np.multiply(kmat,innov)

    # update the ensemble members using the square-root approach
    beta = 1./(1. + np.sqrt(ob_err/(varye+ob_err)))
    kmat = np.multiply(beta,kmat)
    ye   = np.array(ye)[np.newaxis]
    kmat = np.array(kmat)[np.newaxis]
    Xap  = Xbp - np.dot(kmat.T, ye)

    # full state
    Xa = np.add(xam[:,None], Xap)

    # if masked array, making sure that fill_value = nan in the new array 
    if np.ma.isMaskedArray(Xa): np.ma.set_fill_value(Xa, np.nan)

    
    # Return the full state
    return Xa



def Kalman_optimal(Y,vR,Ye,Xb,nsvs=None,transform_only=False,verbose=False):
    """
    Y: observation vector (p x 1)
    vR: observation error variance vector (p x 1)
    Ye: prior-estimated observation vector (p x n)
    Xbp: prior ensemble perturbation matrix (m x n) 

    Originator:

    Greg Hakim
    University of Washington
    26 February 2018

    Modifications:
    11 April 2018: Fixed bug in handling singular value matrix (rectangular, not square)
    """    
    if verbose:
        print('\n all-at-once solve...\n')

    begin_time = time()

    nobs = Ye.shape[0]
    nens = Ye.shape[1]
    ndof = np.min([nobs,nens])
    
    if verbose:
        print('number of obs: '+str(nobs))
        print('number of ensemble members: '+str(nens))
        
    # ensemble prior mean and perturbations
    xbm = Xb.mean(axis=1)
    #Xbp = Xb - Xb.mean(axis=1,keepdims=True)
    Xbp = np.subtract(Xb,xbm[:,None])  # "None" means replicate in this dimension

    R = np.diag(vR)
    Risr = np.diag(1./np.sqrt(vR))
    # (suffix key: m=ensemble mean, p=perturbation from ensemble mean; f=full value)
    # keepdims=True needed for broadcasting to work; (p,1) shape rather than (p,)
    Yem = Ye.mean(axis=1,keepdims=True)
    Yep = Ye - Yem
    Htp = np.dot(Risr,Yep)/np.sqrt(nens-1)
    Htm = np.dot(Risr,Yem)
    Yt = np.dot(Risr,Y)
    # numpy svd quirk: V is actually V^T!
    U,s,V = np.linalg.svd(Htp,full_matrices=True)
    if not nsvs:
        nsvs = len(s) - 1  #TODO: This line makes it differ from the other method.  Should this be "nsvs = len(s)" instead?
    if verbose:
        print('ndof :'+str(ndof))
        print('U :'+str(U.shape))
        print('s :'+str(s.shape))
        print('V :'+str(V.shape))
        print('recontructing using '+ str(nsvs) + ' singular values')
        
    innov = np.dot(U.T,Yt-np.squeeze(Htm))
    # Kalman gain
    Kpre = s[0:nsvs]/(s[0:nsvs]*s[0:nsvs] + 1)
    K = np.zeros([nens,nobs])
    np.fill_diagonal(K,Kpre)
    # ensemble-mean analysis increment in transformed space 
    xhatinc = np.dot(K,innov)
    # ensemble-mean analysis increment in the transformed ensemble space
    xtinc = np.dot(V.T,xhatinc)/np.sqrt(nens-1)
    if transform_only:
        xam = []
        Xap = []
    else:
        # ensemble-mean analysis increment in the original space
        xinc = np.dot(Xbp,xtinc)
        # ensemble mean analysis in the original space
        xam = xbm + xinc

        # transform the ensemble perturbations
        lam = np.zeros([nobs,nens])
        np.fill_diagonal(lam,s[0:nsvs])
        tmp = np.linalg.inv(np.dot(lam,lam.T) + np.identity(nobs))
        sigsq = np.identity(nens) - np.dot(np.dot(lam.T,tmp),lam)
        sig = np.sqrt(sigsq)
        T = np.dot(V.T,sig)
        Xap = np.dot(Xbp,T)    
        # perturbations must have zero mean
        #Xap = Xap - Xap.mean(axis=1,keepdims=True)
        if verbose: print('min s:',np.min(s))
    elapsed_time = time() - begin_time
    if verbose:
        print('shape of U: ' + str(U.shape))
        print('shape of s: ' + str(s.shape))
        print('shape of V: ' + str(V.shape))
        print('-----------------------------------------------------')
        print('completed in ' + str(elapsed_time) + ' seconds')
        print('-----------------------------------------------------')

    readme = '''
    The SVD dictionary contains the SVD matrices U,s,V where V 
    is the transpose of what numpy returns. xtinc is the ensemble-mean
    analysis increment in the intermediate space; *any* state variable 
    can be reconstructed from this matrix.
    '''
    SVD = {'U':U,'s':s,'V':np.transpose(V),'xtinc':xtinc,'readme':readme}
    return xam,Xap,SVD


#========================================================================================== 
#
#========================================================================================== 

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km




def cov_localization(locRad, proxy_lat, proxy_lon, X_coords):
    """

    Originator: R. Tardif, 
                Dept. Atmos. Sciences, Univ. of Washington
    -----------------------------------------------------------------
     Inputs:
        locRad : Localization radius (distance in km beyond which cov are forced to zero)
             Y : Proxy object, needed to get ob site lat/lon (to calculate distances w.r.t. grid pts
             X : Prior object, needed to get state vector info. 
      X_coords : Array containing geographic location information of state vector elements

     Output:
        covLoc : Localization vector (weights) applied to ensemble covariance estimates.
                 Dims = (Nx x 1), with Nx the dimension of the state vector.

     Note: Uses the Gaspari-Cohn localization function.

    """

    # declare the localization array, filled with ones to start with (as in no localization)
    stateVectDim = X_coords.shape[0]
    covLoc = np.ones(shape=[stateVectDim],dtype=np.float64)

    # Mask to identify elements of state vector that are "localizeable"
    # i.e. fields with (lat,lon)
    localizeable = covLoc == 1. # Initialize as True
    
    #for var in X.trunc_state_info.keys():
    #    [var_state_pos_begin,var_state_pos_end] =  X.trunc_state_info[var]['pos']
    #    # if variable is not a field with lats & lons, tag localizeable as False
    #    if X.trunc_state_info[var]['spacecoords'] != ('lat', 'lon'):
    #        localizeable[var_state_pos_begin:var_state_pos_end+1] = False

    # array of distances between state vector elements & proxy site
    # initialized as zeros: this is important!
    dists = np.zeros(shape=[stateVectDim])

    # geographic location of proxy site
    site_lat = proxy_lat
    site_lon = proxy_lon
    # geographic locations of elements of state vector
    X_lon = X_coords[:,1]
    X_lat = X_coords[:,0]

    # calculate distances for elements tagged as "localizeable". 
    dists[localizeable] = np.array(haversine(site_lon, site_lat,
                                                       X_lon[localizeable],
                                                       X_lat[localizeable]),dtype=np.float64)

    # those not "localizeable" are assigned with a disdtance of "nan"
    # so these elements will not be included in the indexing
    # according to distances (see below)
    dists[~localizeable] = np.nan
    
    # Some transformation to variables used in calculating localization weights
    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;
    
    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    # values for distances very near the localization radius
    # TODO: revisit calculations to minimize round-off errors
    covLoc[covLoc < 0.0] = 0.0

    
    return covLoc


def generate_latlon(nlats, nlons, include_endpts=False,
                    lat_bnd=(-90,90), lon_bnd=(0, 360)):
    """
    Generate regularly spaced latitude and longitude arrays where each point 
    is the center of the respective grid cell.
    
    Parameters
    ----------
    nlats: int
        Number of latitude points
    nlons: int
        Number of longitude points
    lat_bnd: tuple(float), optional
        Bounding latitudes for gridcell edges (not centers).  Accepts values 
        in range of [-90, 90].
    lon_bnd: tuple(float), optional
        Bounding longitudes for gridcell edges (not centers).  Accepts values 
        in range of [-180, 360].
    include_endpts: bool
        Include the poles in the latitude array.  

    Returns
    -------
    lat_center_2d:
        Array of central latitide points (nlat x nlon)
    lon_center_2d:
        Array of central longitude points (nlat x nlon)
    lat_corner:
        Array of latitude boundaries for all grid cells (nlat+1)
    lon_corner:
        Array of longitude boundaries for all grid cells (nlon+1)   

    """
    
    if len(lat_bnd) != 2 or len(lon_bnd) != 2:
        raise ValueError('Bound tuples must be of length 2')
    if np.any(np.diff(lat_bnd) < 0) or np.any(np.diff(lon_bnd) < 0):
        raise ValueError('Lower bounds must be less than upper bounds.')
    if np.any(abs(np.array(lat_bnd)) > 90):
        raise ValueError('Latitude bounds must be between -90 and 90')
    if np.any(abs(np.diff(lon_bnd)) > 360):
        raise ValueError('Longitude bound difference must not exceed 360')
    if np.any(np.array(lon_bnd) < -180) or np.any(np.array(lon_bnd) > 360):
        raise ValueError('Longitude bounds must be between -180 and 360')

    lon_center = np.linspace(lon_bnd[0], lon_bnd[1], nlons, endpoint=False)

    if include_endpts:
        lat_center = np.linspace(lat_bnd[0], lat_bnd[1], nlats)
    else:
        tmp = np.linspace(lat_bnd[0], lat_bnd[1], nlats+1)
        lat_center = (tmp[:-1] + tmp[1:]) / 2.

    lon_center_2d, lat_center_2d = np.meshgrid(lon_center, lat_center)
    lat_corner, lon_corner = calculate_latlon_bnds(lat_center, lon_center)

    return lat_center_2d, lon_center_2d, lat_corner, lon_corner


def calculate_latlon_bnds(lats, lons):
    """
    Calculate the bounds for regularly gridded lats and lons.
    
    Parameters
    ----------
    lats: ndarray
        Regularly spaced latitudes.  Must be 1-dimensional and monotonically 
        increase with index.
    lons:  ndarray
        Regularly spaced longitudes.  Must be 1-dimensional and monotonically 
        increase with index.

    Returns
    -------
    lat_bnds:
        Array of latitude boundaries for each input latitude of length 
        len(lats)+1.
    lon_bnds:
        Array of longitude boundaries for each input longitude of length
        len(lons)+1.

    """
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError('Expected 1D-array for input lats and lons.')
    if np.any(np.diff(lats) < 0) or np.any(np.diff(lons) < 0):
        raise ValueError('Expected monotonic value increase with index for '
                         'input latitudes and longitudes')

    # Note: assumes lats are monotonic increase with index
    dlat = abs(lats[1] - lats[0]) / 2.
    dlon = abs(lons[1] - lons[0]) / 2.

    # Check that inputs are regularly spaced
    lat_space = np.diff(lats)
    lon_space = np.diff(lons)

    lat_bnds = np.zeros(len(lats)+1)
    lon_bnds = np.zeros(len(lons)+1)

    lat_bnds[1:-1] = lats[:-1] + lat_space/2.
    lat_bnds[0] = lats[0] - lat_space[0]/2.
    lat_bnds[-1] = lats[-1] + lat_space[-1]/2.

    if lat_bnds[0] < -90:
        lat_bnds[0] = -90.
    if lat_bnds[-1] > 90:
        lat_bnds[-1] = 90.

    lon_bnds[1:-1] = lons[:-1] + lon_space/2.
    lon_bnds[0] = lons[0] - lon_space[0]/2.
    lon_bnds[-1] = lons[-1] + lon_space[-1]/2.

    return lat_bnds, lon_bnds


def lon_lat_to_cartesian(lon, lat, R=6371.):
    """

    """
    
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)

    return x, y, z


def regrid_simple(Nens,X,X_coords,ind_lat,ind_lon,ntrunc):
    """
    Truncate lat,lon grid to another resolution using local distance-weighted 
    averages. 

    Inputs:
    Nens            : number of ensemble members
    X               : data array of shape (nlat*nlon,Nens) 
    X_coords        : array of lat-lon coordinates of variable contained in X
                      w/ shape (nlat*nlon,2)
    ind_lat         : array index (column) at which X_ccords contains latitudes
    ind_lon         : array index (column) at which X_ccords contains longitudes
    ntrunc          : triangular truncation (e.g., use 42 for T42)

    Outputs :
    lat_new : 2D latitude array on the new grid (nlat_new,nlon_new)
    lon_new : 2D longitude array on the new grid (nlat_new,nlon_new)
    X_new   : truncated data array of shape (nlat_new*nlon_new, Nens)
    
    Originator: Robert Tardif
                University of Washington
                March 2017
    """
        
    # truncate to a lower resolution grid (triangular truncation)
    ifix = np.remainder(ntrunc,2.0).astype(int)
    nlat_new = ntrunc + ifix
    nlon_new = int(nlat_new*1.5)

    # create new lat,lon grid arrays
    # Note: AP - According to github.com/jswhit/pyspharm documentation the
    #  latitudes will not include the equator or poles when nlats is even.
    # TODO: Decide whether this should be tied to spherical regridding grids
    if nlat_new % 2 == 0:
        include_poles = False
    else:
        include_poles = True

    lat_new, lon_new, _, _ = generate_latlon(nlat_new, nlon_new,
                                             include_endpts=include_poles)

    # cartesian coords of target grid
    xt,yt,zt = lon_lat_to_cartesian(lon_new.flatten(), lat_new.flatten())

    # cartesian coords of source grid
    lats = X_coords[:, ind_lat]
    lons = X_coords[:, ind_lon]
    xs,ys,zs = lon_lat_to_cartesian(lons, lats)

    # cKDtree object of source grid
    tree = cKDTree(list(zip(xs,ys,zs)))

    # inverse distance weighting (N pts)
    N = 20
    fracvalid = 0.7
    d, inds = tree.query(list(zip(xt,yt,zt)), k=N)
    L = 200.
    w = np.exp(-np.square(d)/np.square(L))

    # transform each ensemble member, one at a time
    X_new = np.zeros([nlat_new*nlon_new,Nens])
    X_new[:] = np.nan
    for k in range(Nens):
        tmp = np.ma.masked_invalid(X[:,k][inds])
        mask = tmp.mask

        # apply tmp mask to weights array
        w = np.ma.masked_where(np.ma.getmask(tmp),w)
        
        # compute weighted-average of surrounding data
        datagrid = np.sum(w*tmp, axis=1)/np.sum(w, axis=1)

        # keep track of valid data involved in averges  
        nbvalid = np.sum(~mask,axis=1)
        nonvalid = np.where(nbvalid < int(fracvalid*N))[0]

        # make sure to mask grid points where too few valid data were used
        datagrid[nonvalid] = np.nan
        X_new[:,k] = datagrid

    # make sure a masked array is returned, if at
    # least one invalid data is found
    if np.isnan(X_new).any():
        X_new = np.ma.masked_invalid(X_new)
    
    return X_new,lat_new,lon_new

