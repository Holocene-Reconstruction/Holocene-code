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

proxy_dir <- 'C:/Users/erbm/Documents/data_climate/data_paleoclimate/proxies/dropbox/'
da_dir    <- 'C:/Users/erbm/Documents/data_climate/data_assimilation/proxies/ecoclimate/'

# LOAD PROXIES =================================================================

# Load the proxy data
data_date <- '2026-07-16'
proxies_all <- readRDS(paste0(proxy_dir,'proxy_ts_',data_date,'.rds'))

# Extract the proxy time series
all_ts <- extractTs(proxies_all)

# MAKE A MAP ===================================================================

# Make a map
#mapLipd(proxies_all, global = TRUE, size = 2) +
#  ggtitle("Proxy records from dropbox")
#ggsave(paste0(proxy_dir,'map_proxies.png'),width=12,height=6)

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

selected <- metadata_all |> 
  filter(datasetname == "BCTempComp.Gavin.2013")

# NA values disrupt the filtering later. Replace all NA values with "NA"
metadata_all <- metadata_all %>% 
  replace(is.na(.), "NA")

# Create a joined field with the primary values, for comparison later
#metadata_all <- metadata_all %>% 
#  mutate(primary1_primary2 = paste0(primary1,'_',primary2))

# FILTER DATA 1 ================================================================

# Set records to remove. Duplicates were identified in another script
#TODO: Update this to refer to a QC sheet
records_to_remove <- c(
  "WEB33ffbfb3",  # Appears to be erroneous
  "LPD5627de74",  # Duplicate with WEB4ef1e8b8
  "WEB33ec2581",  # Duplicate with S2LRE5ZXEOA0gV
  "WEB67e3006e",  # Duplicate with S2LR9DSUVgi2FG
  "GH3c165b1c",   # Notes say to defer to pdRsQ687U7RBI0qSdyE
  "GH20782a02",   # Notes say to defer to pdRPXtR2BTqSS5KCkTm
  "GHeca74745",   # Notes say to defer to pdRkoO11xxFGVWJeL62
  "PYT4E9JNK09",  # Summer version of annual PYTBX6EGUAD
  "PYTKI3J0T5T",  # Summer version of annual PYT9Y0BB33N
  "PYTGVVDEA4T",  # Summer version of annual PYT93BAHDDE
  "R0TCCue9rMw",  # Summer version of annual R7NCYvaRppQ
  "R4IqlneHg0B",  # Winter version of annual RO0fn3HkrCf
  "RUSsCExepnT"   # Summer version of annual RO0fn3HkrCf
  )

# Find records which meet the given criteria
ind_selected <- which(
  
  # Record has values
  metadata_all$has_values
  
  # Record has age or year
  & (metadata_all$has_age | metadata_all$has_year)
  
  # neither paleoData_isPrimary or paleoData_primaryTimeseries is FALSE
  # Note: There are two primary fields. If either are false, do not use them.
  & (metadata_all$primary1 != "FALSE" & metadata_all$primary2 != "FALSE")
  
  # Record is in selected region
  & between(metadata_all$lat,0,85) & between(metadata_all$lon,-180,-10)
  
  # archiveType is not Midden
  & (metadata_all$archivetype != "Midden")
  
  # paleoData_proxy is not pollen
  & (metadata_all$proxytype != "pollen")
  
  # interpretation1_variable is temperature or precipitation
  & metadata_all$interpvar %in% c("temperature","effectivePrecipitation","precipitation")
  
  # Remove some records
  & !metadata_all$tsid %in% records_to_remove
  
)

# Get the selected records and metadata
ts_selected <- all_ts[ind_selected]
metadata_selected <- metadata_all[ind_selected,]

# FILTER DATA 2 ================================================================

# Loop through all records
record_length_all <- c()
i <- 1
for (i in 1:length(ts_selected)) {
  #
  # If ages are missing, create an age column using the year column
  record_names <- names(ts_selected[[i]])
  if (!"age" %in% record_names) {
    message('NOTE: Adding age values from year values for record ',i)
    ts_selected[[i]]$age <- 1950 - ts_selected[[i]]$year
  }
  #
  # Get data and ages
  record_data <- ts_selected[[i]]$paleoData_values
  record_ages <- ts_selected[[i]]$age
  #
  # Compute record length
  ind_valid <- is.finite(record_data) & is.finite(record_ages)
  record_data_valid <- record_data[ind_valid]
  record_ages_valid <- record_ages[ind_valid]
  record_length <- max(record_ages_valid) - min(record_ages_valid)
  record_length_all[[i]] <- record_length
  #
}

# Add to dataframe
metadata_selected$record_length <- record_length_all

# Find records which meet the given criteria
ind_selected_step2 <- which(record_length_all >= 2500)

# Get the selected records and metadata
ts_selected <- ts_selected[ind_selected_step2]
metadata_selected <- metadata_selected[ind_selected_step2,]

# SAVE RECORDS =================================================================

# Save filtered data
#saveRDS(ts_selected,file=paste0(da_dir,'ecoclimate_selected_ts_',data_date,'.rds'))
saveRDS(metadata_selected,file=paste0(da_dir,'ecoclimate_selected_metadata_',data_date,'.rds'))
write.csv(metadata_selected$tsid,file=paste0(da_dir,'ecoclimate_selected_metadata_',data_date,'.csv'))
