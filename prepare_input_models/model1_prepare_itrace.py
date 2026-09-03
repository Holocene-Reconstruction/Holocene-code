#==============================================================================
# Make some plots using the new iTRACE Holocene simulation.
#    author: Michael Erb
#==============================================================================

import numpy as np
import xarray as xr

# Select variable to process
#var_selected = "tas"
#var_selected = "precip"
var_selected = "d18Op_unweighted"
#var_selected = "d18Op_weighted"


#%% LOAD DATA

# Directories
data_dir_holocene  = 'P:/data_paleoclimate/models/itrace_holocene/'
data_dir_deglacial = 'P:/data_paleoclimate/models/itrace/'
output_dir         = "P:/data_paleoclimate/models/itrace_combined/"

# Load data: Holocene
if   var_selected == "tas":              file_txt = "TREFHT";    var_txt = "TREFHT"
elif var_selected == "precip":           file_txt = "precp";     var_txt = "precp"
elif var_selected == "d18Op_unweighted": file_txt = "d18Op_uw";  var_txt = "d18O"
elif var_selected == "d18Op_weighted":   file_txt = "d18Op_wgt"; var_txt = "d18Ow"
handle_holocene = xr.open_dataset(data_dir_holocene+'atm-surface/itrace.11Ka-0Ka.atm.'+file_txt+'.nc',decode_times=False)
var_holocene_mam = handle_holocene[var_txt+'_MAM'].values
var_holocene_jja = handle_holocene[var_txt+'_JJA'].values
var_holocene_son = handle_holocene[var_txt+'_SON'].values
var_holocene_djf = handle_holocene[var_txt+'_DJF'].values
lat_holocene = handle_holocene['lat'].values
lon_holocene = handle_holocene['lon'].values
handle_holocene.close()

# Get ages
age_holocene = np.genfromtxt(data_dir_holocene+'time_values1.txt')
age_holocene = age_holocene + 0.5  # Adjust the ages to better represent the mean of years. Is this right?

# Load data: deglacial
if var_selected == "tas":
    handle_deglacial = xr.open_dataset(data_dir_deglacial+'b.e13.Bi1850C5.f19_g16.all.21_12ka.itrace.ice_ghg_orb_wtr.cam.h0.TREFHT.000101-899912.res_10.nc')
    var_deglacial = handle_deglacial['TREFHT'].values
elif var_selected == "precip":
    handle_deglacial  = xr.open_dataset(data_dir_deglacial+'b.e13.Bi1850C5.f19_g16.all.21_12ka.itrace.ice_ghg_orb_wtr.cam.h0.PRECC.000101-899912.res_10.nc')
    handle_deglacial2 = xr.open_dataset(data_dir_deglacial+'b.e13.Bi1850C5.f19_g16.all.21_12ka.itrace.ice_ghg_orb_wtr.cam.h0.PRECL.000101-899912.res_10.nc')
    precc_deglacial = handle_deglacial['PRECC'].values
    precl_deglacial = handle_deglacial2['PRECL'].values
    var_deglacial = precc_deglacial + precl_deglacial
    handle_deglacial2.close()
elif var_selected in ["d18Op_unweighted","d18Op_weighted"]:
    #
    # Load variables
    variables_to_load = ['PRECC','PRECL','PRECRC_H218Or','PRECRL_H218OR','PRECSC_H218Os','PRECSL_H218OS','PRECRC_H216Or','PRECRL_H216OR','PRECSC_H216Os','PRECSL_H216OS']
    var_deglacial_all = {}
    for variable in variables_to_load:
        handle_deglacial = xr.open_dataset(data_dir_deglacial+'b.e13.Bi1850C5.f19_g16.all.21_12ka.itrace.ice_ghg_orb_wtr.cam.h0.'+variable+'.000101-899912.res_10.nc')
        var_deglacial_all[variable] = handle_deglacial[variable].values
        handle_deglacial.close()
    #
    # Compute totals from the different components
    H218O_total = var_deglacial_all['PRECRC_H218Or'] + var_deglacial_all['PRECRL_H218OR'] + var_deglacial_all['PRECSC_H218Os'] + var_deglacial_all['PRECSL_H218OS']
    H216O_total = var_deglacial_all['PRECRC_H216Or'] + var_deglacial_all['PRECRL_H216OR'] + var_deglacial_all['PRECSC_H216Os'] + var_deglacial_all['PRECSL_H216OS']
    #precip_total = var_deglacial_all['PRECC'] + var_deglacial_all['PRECL']
    #
    # Compute d18O
    d18O = ((H218O_total / H216O_total) - 1) * 1000
    #
    # Weight by precipitation
    d18O_weighted = d18O * np.nan  # TODO: How do I do this? (Set to nan for now)
    #
    # Get variable of interest
    if   var_selected == "d18Op_unweighted": var_deglacial = d18O
    elif var_selected == "d18Op_weighted":   var_deglacial = d18O_weighted

lat_deglacial  = handle_deglacial['lat'].values
lon_deglacial  = handle_deglacial['lon'].values
time_deglacial = handle_deglacial['time'].values
handle_deglacial.close()
age_deglacial = 20000 - time_deglacial

# Compare lats and lons
print('iTRACE:          ',min(lat_deglacial),max(lat_deglacial),min(lon_deglacial),max(lon_deglacial))
print('iTRACE-Holocene: ',min(lat_holocene), max(lat_holocene), min(lon_holocene), max(lon_holocene))
if max(lat_deglacial - lat_holocene) > 0: print("Warning: differences in lat")
if max(lon_deglacial - lon_holocene) > 0: print("Warning: differences in lon")


#%% CREATE A SORT-OF MONTHLY FILE UNTIL I CAN GET A REAL ONE

# Set up
n_time = var_holocene_mam.shape[0]
n_lat  = var_holocene_mam.shape[1]
n_lon  = var_holocene_mam.shape[2]
var_holocene_pseudomonthly = np.zeros((n_time,12,n_lat,n_lon)); var_holocene_pseudomonthly[:] = np.nan

# Copy the seasonal data into the monthly variable
var_holocene_pseudomonthly[:,[0,1],:,:]    = var_holocene_djf[:,np.newaxis,:,:]
var_holocene_pseudomonthly[:,[2,3,4],:,:]  = var_holocene_mam[:,np.newaxis,:,:]
var_holocene_pseudomonthly[:,[5,6,7],:,:]  = var_holocene_jja[:,np.newaxis,:,:]
var_holocene_pseudomonthly[:,[8,9,10],:,:] = var_holocene_son[:,np.newaxis,:,:]
var_holocene_pseudomonthly[:,[11],:,:]     = var_holocene_djf[:,np.newaxis,:,:]


#%% COMBINE THE TWO PARTS

# Convert units from K to C
if var_selected == "tas":
    unit_txt = "degC"
    var_holocene_pseudomonthly = var_holocene_pseudomonthly - 273.15  # K to C
    var_deglacial = var_deglacial - 273.15  # K to C
elif var_selected == "precip":
    unit_txt = "mm/day"
    var_deglacial = var_deglacial * 1000*60*60*24  # m/s to mm/day
elif var_selected in ["d18Op_unweighted","d18Op_weighted"]:
    unit_txt = "permil"

# Combine the older and newer segments
var_combined = np.concatenate((var_deglacial,var_holocene_pseudomonthly),axis=0)
age_combined = np.concatenate((age_deglacial,age_holocene),axis=0)
lat_combined = lat_deglacial
lon_combined = lon_deglacial


#%% SAVE MODEL OUTPUT

# Construct the variable to save
data_to_save = xr.Dataset(
    {
        var_selected: (["age","month","lat","lon"],var_combined,{"units":unit_txt}),
    },
    coords = {
        "age":   (["age"],  age_combined,    {"units":"yr BP (ref 1950)"}),
        'month': (["month"],np.arange(1,13),{"units":"month_number"}),
        "lat":   (["lat"],  lat_combined,    {"units":"degrees_north"}),
        "lon":   (["lon"],  lon_combined,    {"units":"degrees_east"}),
        },
    attrs={
        'description':'iTRACE and iTRACE-Holocene, combined. "Months" in the Holocene portion (11-0 ka) are seasonal values repeated into months',
    }
)

# Save the data
data_to_save.to_netcdf(output_dir+"itrace.19999-0BP."+var_selected+".timeres_10.nc")

