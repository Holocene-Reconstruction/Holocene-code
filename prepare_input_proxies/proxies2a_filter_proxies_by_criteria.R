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

# NA values disrupt the filtering later. Replace all NA values with "NA"
metadata_all <- metadata_all %>% 
  replace(is.na(.), "NA")

# Create a joined field with the primary values, for comparison later
#metadata_all <- metadata_all %>% 
#  mutate(primary1_primary2 = paste0(primary1,'_',primary2))

# FILTER DATA 1 ================================================================

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

)

# Get the selected records and metadata
ts_selected <- all_ts[ind_selected]
metadata_selected <- metadata_all[ind_selected,]

# FILTER DATA 2 ================================================================

# Loop through all records
record_length_all <- c()
n_data_all        <- c()
n_unique_all      <- c()
i <- 683
for (i in 1:length(ts_selected)) {
  
  # If ages are missing, create an age column using the year column
  record_names <- names(ts_selected[[i]])
  if (!"age" %in% record_names) {
    message('NOTE: Adding age values from year values for record ',i)
    ts_selected[[i]]$age <- 1950 - ts_selected[[i]]$year
  }
  
  # Get data and ages
  record_data <- ts_selected[[i]]$paleoData_values
  record_ages <- ts_selected[[i]]$age
  
  # Compute record length
  ind_valid <- is.finite(record_data) & is.finite(record_ages)
  record_data_valid <- record_data[ind_valid]
  record_ages_valid <- record_ages[ind_valid]
  record_length <- max(record_ages_valid) - min(record_ages_valid)
  record_length_all[i] <- record_length

  # Compute number of unique data points within 0-21 ka
  data_in_period <- record_data[(record_ages <= 21000) & (record_ages >= 0)]
  data_in_period <- data_in_period[is.finite(data_in_period)]
  n_data   <- length(data_in_period)
  n_unique <- length(unique(data_in_period))
  n_data_all[i]   <- n_data
  n_unique_all[i] <- n_unique
  
}

# Add to dataframe
metadata_selected$record_length <- record_length_all
metadata_selected$n_data        <- n_data_all
metadata_selected$n_unique      <- n_unique_all

# Find records which meet the given criteria
ind_selected_step2 <- which(
  
  # Record length is at least 2500 years
  record_length_all >= 2500
  
  # Records have at least 3 unique data points
  & n_unique_all >= 3

  # Records have at least 5 data points in total
  & n_data_all >= 5
  
)

# Get the selected records and metadata
ts_selected <- ts_selected[ind_selected_step2]
metadata_selected <- metadata_selected[ind_selected_step2,]

# SAVE RECORDS =================================================================

# Save filtered data
#saveRDS(ts_selected,             file=paste0(da_dir,'ecoclimate_selected_ts_',data_date,'_by_criteria.rds'))
#saveRDS(metadata_selected,       file=paste0(da_dir,'ecoclimate_selected_metadata_',data_date,'_by_criteria.rds'))
#write.csv(metadata_selected$tsid,file=paste0(da_dir,'ecoclimate_selected_metadata_',data_date,'_by_criteria.csv'))

library(ggpubr)

# CHRIS
i=30
df<-data.frame()
for (i in 1:length(ts_selected)){
  ts= ts_selected[[i]]
  if (sum(is.finite(ts$paleoData_values))<3){next}
  if (length(unique(ts$paleoData_values))<3){next}
  print(i)
  if (is.null(ts$interpretation1_interpDirection)){
    interpDir <- ""
  }else if (ts$interpretation1_interpDirection %in% c("negative","-1")){
    interpDir <- "*-1"
    ts$vals<-ts$vals*-1
  } else{
    interpDir <- ""
  }
  if (is.null(ts$interpretation1_direction)){
    interpDir <- ""
  }else if (ts$interpretation1_direction %in% c("negative","-1")){
    interpDir <- "*-1"
    ts$vals<-ts$vals*-1
  } else{
    interpDir <- ""
  }
  ts_bin <- data.frame(age=seq(0,21000,100),
                       vals=compositeR::simpleBinTs(ts,binvec=seq(-50,21050,100),spread=T))#,spreadMax=500))
  ts_percent = data.frame(age=ts_bin$age,vals=ts_bin$vals) %>%
    mutate(vals = percent_rank(vals) * 100)

  y1_range <- range(ts_bin$vals, na.rm = TRUE)
  y2_range <- range(ts_percent$vals, na.rm = TRUE)
  scale_factor <- diff(y1_range) / diff(y2_range)
  offset <- y1_range[1] - y2_range[1] * scale_factor

  idx <- (is.finite(ts_bin$vals) & is.finite(ts_percent$vals))
  corcoef<-sprintf("%.3f",cor(ts_bin$vals[idx],ts_percent$vals[idx]))
  a <- ggplot()+
    geom_line(aes(x=ts_bin$age[is.finite(ts_bin$vals)],y=ts_bin$vals[is.finite(ts_bin$vals)],color='binned data'))+
    geom_point(aes(x=ts_bin$age[is.finite(ts_bin$vals)],y=ts_bin$vals[is.finite(ts_bin$vals)],color='binned data'))+
    geom_line(aes(x=ts_percent$age, y=ts_percent$vals * scale_factor + offset, color='percentile'))+
    geom_point(aes(x=ts_percent$age,y=ts_percent$vals * scale_factor + offset, color='percentile'))+
    geom_point(aes(x=ts$age,y=ts$paleoData_values),color='black',fill=NA,shape=21)+

    #
    scale_y_continuous(
      name = paste0(ts$paleoData_variableName,' - ',ts$interpretation1_variable,' (',ts$paleoData_units,')'),
      sec.axis = sec_axis(
        ~ (. - offset) / scale_factor,name = "Percentile"),#,limits=c(0,100))
      )+
    scale_x_reverse(name=paste0('age (',ts$ageUnits,')'), )+#lim=c(21000,0))+
    labs(title=paste(ts$dataSetName,ts$paleoData_TSid,ts$paleoData_proxy,sep=' ///// '),
         subtitle=paste('correlation coefficient:',corcoef))+
    theme_bw()

  b <- ggplot() +
    geom_histogram(aes(x = ts_bin$vals, color='binned data'), fill = NA, , bins=30) +
    geom_histogram(aes(x = ts_percent$vals* scale_factor + offset, color='percentile'),  fill = NA,linetype='dashed', bins=30)+
    labs(x=paste0(ts$paleoData_variableName,' - ',ts$interpretation1_variable,' (',ts$paleoData_units,interpDir,')'),
         title='Histrogram of values')+
    scale_x_continuous(
      name = paste0(ts$paleoData_variableName,' - ',ts$interpretation1_variable,' (',ts$paleoData_units,interpDir,')'),
      sec.axis = sec_axis(
        ~ (. - offset) / scale_factor,name = "Percentile"),#,limits=c(0,100))
    )+
    theme_bw() +
    theme(legend.position="none")

  plt<-ggarrange(a,b, ncol = 2, nrow = 1, common.legend = TRUE, legend = "bottom",widths=c(0.66,0.34))
  if (ts$interpretation1_variable=='temperature'){
    path <- file.path(da_dir,'PercentileCorr','temp')
  }else{
    path <- file.path(da_dir,'PercentileCorr','hydro')
  }
  fn <- paste0(paste(corcoef,ts$dataSetName,ts$paleoData_TSid,sep='_'),'.png')
  ggsave(filename=fn,path=path,plot=plt,width=8,height=4,units='in')

  if (is.null(ts$paleoData_proxy)){
    proxy<-'NA'
  }else{
    proxy<- ts$paleoData_proxy
  }

  df_i <- data.frame(tsid=ts$paleoData_TSid,
                     proxy=proxy,
                     interp=ts$interpretation1_variable,
                     correlation=corcoef)
  df<-rbind(df,df_i)
}

