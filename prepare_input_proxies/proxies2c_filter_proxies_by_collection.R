#===============================================================================
# This package loads the proxy database so it can be explored and filtered.
# Code references:
#  - https://nickmckay.org/lipdR/
#  - https://nickmckay.org/GeoChronR/articles/TsFilteringAndMapping.html
# author: Michael Erb
#===============================================================================

library(lipdR)
library(tidyverse)

proxy_dir <- 'P:/data_paleoclimate/data_assimilation/proxies/'

# LOAD PROXIES =================================================================

# Load the proxy data
data_date <- '2026-09-11'
proxies_all <- readRDS(paste0(proxy_dir,'proxy_ts_',data_date,'.rds'))

# Extract the proxy time series
all_ts <- extractTs(proxies_all)
#test <- all_ts[1][[1]]

# GET RECORDS IN COMPILATION ===================================================

compilation_txt = "NAm21k-noPollen"

counter <- 1
ind_selected <- c()
for (i in 1:length(all_ts)) {
  record_selected <- all_ts[i][[1]]
  if ((record_selected["inCompilation1_compilationName"] == compilation_txt) |
      (record_selected["inCompilation2_compilationName"] == compilation_txt) |
      (record_selected["inCompilation3_compilationName"] == compilation_txt) |
      (record_selected["inCompilation4_compilationName"] == compilation_txt) |
      (record_selected["inCompilation5_compilationName"] == compilation_txt) |
      (record_selected["inCompilation6_compilationName"] == compilation_txt)) {
    ind_selected[counter] <- i
    counter <- counter + 1
  }
}
ts_selected <- all_ts[ind_selected]


# CREATE DATAFRAME WITH METADATA ===============================================

# Create a dataframe with metadata
metadata_selected <- data.frame(
  tsid        = pullTsVariable(ts_selected,"paleoData_TSid"),
  datasetname = pullTsVariable(ts_selected,"dataSetName"),
  primary1    = pullTsVariable(ts_selected,"paleoData_isPrimary"),
  primary2    = pullTsVariable(ts_selected,"paleoData_primaryTimeseries"),
  primary3    = pullTsVariable(ts_selected,"paleoData_useInGlobalTemperatureAnalysis"),
  variable    = pullTsVariable(ts_selected,"paleoData_variableName"),
  lat         = pullTsVariable(ts_selected,"geo_latitude"),
  lon         = pullTsVariable(ts_selected,"geo_longitude"),
  archivetype = pullTsVariable(ts_selected,"archiveType"),
  proxytype   = pullTsVariable(ts_selected,"paleoData_proxy"),
  interpvar   = pullTsVariable(ts_selected,"interpretation1_variable"),
  units       = pullTsVariable(ts_selected,"paleoData_units"),
  temp_uncer  = pullTsVariable(ts_selected,"paleoData_temperature12kUncertainty"),
  description = pullTsVariable(ts_selected,"paleoData_description"),
  has_values  = !sapply(pullTsVariable(ts_selected,"paleoData_values"), is.null),
  has_age     = !sapply(pullTsVariable(ts_selected,"age"), is.null),
  has_year    = !sapply(pullTsVariable(ts_selected,"year"), is.null)
)

# NA values disrupt the filtering later. Replace all NA values with "NA"
metadata_selected <- metadata_selected %>% 
  replace(is.na(.), "NA")

# SAVE RECORDS =================================================================

# Save filtered data
saveRDS(ts_selected,      file=paste0(proxy_dir,'ecoclimate_selected_ts_',data_date,'.rds'))
saveRDS(metadata_selected,file=paste0(proxy_dir,'ecoclimate_selected_metadata_',data_date,'.rds'))
