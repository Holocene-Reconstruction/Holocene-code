#===============================================================================
# This package loads the proxy database and stores it in an easier-to-load format.
# The proxy database can be downloaded from:
# https://www.dropbox.com/scl/fo/0fg5l38liybjnsv1wwb9z/ACpHwueg_eNtbIJ1OAbMXLk?rlkey=rikg5mo70sbvq3ucat9qctb0a&e=3&st=aiywhu91&dl=0
#
# Code references:
#  - https://nickmckay.org/lipdR/
#  - https://nickmckay.org/GeoChronR/articles/TsFilteringAndMapping.html
# author: Michael Erb
#===============================================================================

library(lipdR)
library(geoChronR)

proxy_dir <- 'P:/data_paleoclimate/proxies/dropbox/'

# LOAD PROXIES =================================================================

proxies_all <- readLipd(paste0(proxy_dir,'database/'))
all_ts <- extractTs(proxies_all)

# SAVE DATA ====================================================================

# Save data into a different format
date_today <- as.character(Sys.Date())
saveRDS(proxies_all, file=paste0(proxy_dir,'proxy_ts_',date_today,'.rds'))
