#==============================================================================
# Take a look at the proxy database.
#    author: Michael P. Erb
#    date  : 1/13/2020
#==============================================================================

import numpy as np
import pickle
import lipd


#%% LOAD DATA

# Load the Temp12k proxy metadata
data_dir = '/projects/pd_lab/data/proxies/Holocene/database_v1/'
file_to_open = open(data_dir+'Temp12k1_0_1.pkl','rb')
proxies_all = pickle.load(file_to_open)['D']
file_to_open.close()

# Extract the time series and use only those which are in Temp12k and in units of degC
all_ts = lipd.extractTs(proxies_all)
filtered_ts = lipd.filterTs(all_ts,'paleoData_inCompilation == Temp12k')
filtered_ts = lipd.filterTs(filtered_ts,'paleoData_units == degC')


#%% CALCULATIONS

# Save the interpretation and seasonality metadata
archivetype_all = []
n_proxies = len(filtered_ts)
for i in range(n_proxies):
    archivetype_all.append(filtered_ts[i]['archiveType'])

#archivetype_all = np.array(archivetype_all)
print(np.unique(archivetype_all,return_counts=True))


#%%
selected_proxies = ['GlacierIce','Ice-other','Speleothem']
ind_archives = [i for i, x in enumerate(archivetype_all) if x in selected_proxies]
chosen_archives = np.full((n_proxies),False,dtype=bool)
chosen_archives[ind_archives] = True

