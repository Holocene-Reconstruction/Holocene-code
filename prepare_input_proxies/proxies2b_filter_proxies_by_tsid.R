#===============================================================================
# This package loads the proxy database so it can be explored and filtered.
# Code references:
#  - https://nickmckay.org/lipdR/
#  - https://nickmckay.org/GeoChronR/articles/TsFilteringAndMapping.html
# author: Michael Erb
#===============================================================================

library(lipdR)
library(geoChronR)
library(ggplot2)
library(tidyverse)
library(data.table)
library(writexl)
library(readxl)

proxy_dir <- 'P:/data_paleoclimate/proxies/dropbox/'
da_dir    <- 'P:/data_paleoclimate/data_assimilation/proxies/ecoclimate/'

# LOAD PROXIES =================================================================

# Load the proxy data
data_date <- '2026-08-31'
proxies_all <- readRDS(paste0(proxy_dir,'proxy_ts_',data_date,'.rds'))

# Load the QC sheet
qc_sheet <- read_excel(paste0(da_dir,"qc_sheet/NAm21k-noPollen v.0_3_0 QC sheet.xlsx"),
                       guess_max=10000, sheet="QC")

# Extract the proxy time series
all_ts <- extractTs(proxies_all)

# PRINT METADATA ===============================================================

# A function to print summaries of the selected metadata field
print_counts <- function(selected_ts,var_name) {
  metadata_values <- pullTsVariable(selected_ts,var_name)
  metadata_values_df <- as.data.frame(metadata_values)
  colnames(metadata_values_df) <- c("var")
  count_metadata_values <- metadata_values_df %>% 
    count(var) %>% 
    arrange(desc(n))
  message(var_name,' - Total: ',nrow(metadata_values_df))
  print(count_metadata_values)
}

#print_counts(all_ts,"archiveType")
#print_counts(all_ts,"paleoData_proxy")
#print_counts(all_ts,"paleoData_units")
#print_counts(all_ts,"ageUnits")
#print_counts(all_ts,"paleoData_variableName")
#print_counts(all_ts,"paleoData_longName")
#print_counts(all_ts,"paleoData_summaryStatistic")
#print_counts(all_ts,"paleoData_isPrimary")
#print_counts(all_ts,"paleoData_primaryTimeseries")
#print_counts(all_ts,"interpretation1_direction")  # Note: Alternate version with capital I
#print_counts(all_ts,"interpretation1_scope")      # Note: Alternate version with capital I
#print_counts(all_ts,"interpretation1_seasonality")
#print_counts(all_ts,"interpretation1_seasonalityGeneral")
#print_counts(all_ts,"interpretation1_variable")        # Note: Alternate version with capital I
#print_counts(all_ts,"interpretation1_variableDetail")  # Note: Alternate version with capital I
#print_counts(all_ts,"paleoData_temperature12kUncertainty")
#print_counts(all_ts,"paleoData_description")

# Other potentially useful vars:
# agesPerKyr, maxYear, minYear, geo_latitude,geo_longitude, paleoData_TSid, paleoData_summaryStatistic

# CREATE DATAFRAME WITH METADATA ===============================================

# Create a dataframe with metadata
metadata_all <- data.frame(
  tsid        = pullTsVariable(all_ts,"paleoData_TSid"),
  datasetname = pullTsVariable(all_ts,"dataSetName"),
  primary1    = pullTsVariable(all_ts,"paleoData_isPrimary"),
  primary2    = pullTsVariable(all_ts,"paleoData_primaryTimeseries"),
  primary3    = pullTsVariable(all_ts,"paleoData_useInGlobalTemperatureAnalysis"),
  variable    = pullTsVariable(all_ts,"paleoData_variableName"),
  lat         = pullTsVariable(all_ts,"geo_latitude"),
  lon         = pullTsVariable(all_ts,"geo_longitude"),
  archivetype = pullTsVariable(all_ts,"archiveType"),
  proxytype   = pullTsVariable(all_ts,"paleoData_proxy"),
  interpvar   = pullTsVariable(all_ts,"interpretation1_variable"),
  units       = pullTsVariable(all_ts,"paleoData_units"),
  temp_uncer  = pullTsVariable(all_ts,"paleoData_temperature12kUncertainty"),
  description = pullTsVariable(all_ts,"paleoData_description"),
  has_values  = !sapply(pullTsVariable(all_ts,"paleoData_values"), is.null),
  has_age     = !sapply(pullTsVariable(all_ts,"age"), is.null),
  has_year    = !sapply(pullTsVariable(all_ts,"year"), is.null)
)

# NA values disrupt the filtering later. Replace all NA values with "NA"
metadata_all <- metadata_all %>% 
  replace(is.na(.), "NA")

# FILTER DATA 1 ================================================================

# Get the records to keep from the QC sheet
records_to_keep <- qc_sheet |> 
  filter(inThisCompilation == TRUE) |> 
  select(dataSetName, TSid)

# Find records which meet the given criteria
ind_selected <- which(
  
  # Only keep records approved in the QC sheet
  metadata_all$tsid %in% records_to_keep$TSid
  
)

# Get the selected records and metadata
ts_selected <- all_ts[ind_selected]
metadata_selected <- metadata_all[ind_selected,]

# SAVE RECORDS =================================================================

# Save filtered data
saveRDS(ts_selected,             file=paste0(da_dir,'ecoclimate_selected_ts_',data_date,'.rds'))
saveRDS(metadata_selected,       file=paste0(da_dir,'ecoclimate_selected_metadata_',data_date,'.rds'))
write.csv(metadata_selected$tsid,file=paste0(da_dir,'ecoclimate_selected_metadata_',data_date,'.csv'))
