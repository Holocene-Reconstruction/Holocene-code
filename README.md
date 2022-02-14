# README FOR HOLOCENE RECONSTRUCTION PROJECT
Written by: Michael P. Erb, Contact: michael.erb@nau.edu

## 1. Introduction

Data and models are two methods of exploring past climate.  Data (such as proxy records) provide point data and models simulate climate changes and climate-system interactions.  The goal of this Holocene Reconstruction project is to use data assimilation--a method of combining information from proxy data and model results--to reconstruct climate over the past 12,000 years.

This GitHub repository contains the Holocene Reconstruction code, and this readme file explains how to set it up and use it.

This code and readme are still under development.

## 2. Getting started

The Holocene Reconstruction code is written in Python 3.  The instructions below will help you download the Holocene Reconstruction code, download the necessary data files, install Python 3, and start using the code.

### 2.1. Getting the Holocene Reconstruction code

Clone the Github repository into your Linux environment with the command:

    git clone https://github.com/Holocene-Reconstruction/Holocene-code.git

### 2.2. Getting the necessary data files

The code uses climate model output as well as proxy data files.  To get this data, download the zip file from this link: [data on Google Drive](https://drive.google.com/file/d/1Iqfbpa4mhoIw_ccKYzfljkkTTz47HKzJ/view?usp=sharing)

Put this fine in a convenient place and unzip it using `unzip holocene_da_data.zip`.  It should contain the following subdirectories:

    models/   Model output
    proxies/  Proxy files
    results/  Results of the data assimilation (initially empty)

The data set currently includes only the TraCE-21ka model output and Temp12k proxies.  Other data may be added later.

### 2.3. Installing Python 3 and necessary packages

Make sure that you have Python 3 installed.  If you don't, one option is the package manager Anaconda: [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)

Most of the necessary packages should come with the standard Python 3 installation, installed using `conda create -n python3_da anaconda`.  The only ones you should need to install are xarray, netCDF4, and lipd.  Either install these yourself (note: LiPD can be installed with `pip install LiPD`) or go to your Holocene Reconstruction directory and use the commands below to create a new Python 3 environment with the necessary packages and switch to it:

    conda env create -f environment_da.py
    conda activate python3_da

NOTE: If you have trouble installing the LiPD library, you can use the following workaround:
 1. Clone the github repository at [https://github.com/nickmckay/LiPD-utilities](https://github.com/nickmckay/LiPD-utilities) to a convenient location.
 2. Go to your Holocene Reconstruction directory and open config_default.yml.  Change the "lipd_dir" line near the top to specify the location of the newly created LiPD directory.  If you're not using this workaround, feel free to delete the 'lipd_dir' line in config_default.yml.

### 2.4. First-time setup

Before running the Holocene Reconstruction code for the first time, do the following:
 1. Open config_default.py.
 2. Near the top, change the 'data_dir' directory to the location of the unzipped data from section 2.2 above (i.e., the directory which has the models/, proxies/, and results/ subdirectories.)
 3. Save config_default.py and copy it to config.py.  You can set up new experiments in config.py while keeping the original file for reference.

## 3. Running the code

You should now be ready to run the data assimilation code.  To run the code using the default options, go to your Holocene Reconstruction directory and execute the command:

    python da_main_code.py config.yml

The code will update you on what it's doing as it runs.  The code can take some time to run, so you may want to submit it in the background or use a job scheduler.  If you use the job scheduler Slurm, you can use the run_da.sh file.  Double-check the settings in that file and run it with `sbatch run_da.sh`.

If the reconstruction finishes successfully, output will be saved as a netCDF file in the results/ subdirectory of the 'data_dir' directory.  See section 5 for more details.

## 4. Setting up new experiments

Running the code as-is will use the default settings.  To change settings and run new experiments, change variables in the `config.yml` file, some of which are explained below.

### 4.1. Experimental options

The file `config.yml` contains a variety of settings for running new experiments.  Some of these settings are explained below, and others can be seen by opening the config.yml file.  Not all of these options have been thoroughly tested, so some experimental designs may crash the code.

#### Age range to reconstruct

By default, the Holocene Reconstruction code reconstructs the past 12,000 years.  This is set in the `age_range_to_reconstruct` variable (default: [0,12000]), but can be shortened or lengthened as requested.  The Temp12k proxy database contains proxy data older than 12ka, but the quality of the reconstruction will decline as the number of available proxies diminish further back in time.  Another relevant variable is `reference_period` (default: [0,5000]), which defines the reference period used for each proxy.  Proxies without data in the selected reference period will not be used.

#### Proxies to assimilate

The variable `proxy_datasets_to_assimilate` (default: ['temp12k']) lists the proxy database(s) to be assimilated.  Only Temp12k proxies are supported out-of-the-box.  Other proxy datasets, such as PAGES2k, may be added later.

#### Models to use as the prior

The variable `models_for_prior` (default: ['trace_regrid']) defines the model simulation(s) to be used as the model prior.  Only TraCE-21ka is included in the data set download in section 2.2 above, but HadCM3 will be added soon.

#### The size of the prior 'window'

The model prior consists of a set of model states chosen from the simulation(s) specified above.  By default, the code doesn't use all modeled climate states, but selects a 'window' of time surrounding each year of the reconstruction.  The length of this window, in years, is set in the variable `prior_window` (default: 5010).  A smaller value will allow the prior to change more through time, but too few model states may result in a poorer reconstruction, and vice versa.  To use all climate states from the model prior (resulting in a prior that does not change with time) set this to 'all'.

#### Time resolutions

The Holocene Reconstruction uses a multi-timescale approach to data assimilation, which can assimilate proxies at a variety of timescales.  To change the details of how this works, the variables `time_resolution` (default: 10) and `maximum_resolution` (default: 1000) can be changed.  `time_resolution` defines the base temporal resolution, in years, used in the reconstruction.  Proxy data will be preprocessed at this resolution and the final reconstruction will be output at this resolution.  Few proxies in the Temp12k database have temporal resolution finer than 10 years, so little information should be lost when using this base resolution.  Changing this number should also affect the speed of the data assimilation, but this has not been well-tested.  `maximum_resolution` specifies the maximum temporal resolution, in years, that proxy data points will be treated as.  This option does not affect which proxies are assimilated or not, but only affects the timescale which is used when translating proxy data to the larger climate system; proxy data points which appear to represent longer periods of time will be assumed to represent this amount of time instead.

#### Localization radius

If a localization radius is desired, set it in the `localization_radius` (default: None) variable, in units of meters.  To use a localization radius, the variable `assimate_together` (default: True) should be set to False, or else the localization radius will not be used.  Setting `assimilate_together` to False will significantly slow down the code, since it will assimilate proxies one at a time instead of all together.  However, this must be done for a localization radius to work.

#### Assimilating only a portion of the proxy database

By default, all valid proxies are assimilated.  To assimilate only a portion of the proxy database, set the variable `percent_to_assimilate` (default: 100) to a lower number.

#### Using pseudoproxies

To generate/use pseudoproxies, the string in 'proxy_datasets_to_assimilate' should be given in the form ['pseudo_VAR1_using_VAR2_noise_VAR3'], where:
  VAR1: proxy dataset [e.g. 'temp12k','basicgrid10','basicgrid5']
  VAR2: model [e.g. 'hadcm3','trace','famous']
  VAR3: noise [e.g. 'none','whitesnr05','whiteproxyrmse']
Example: ['pseudo_temp12k_using_hadcm3_noise_whiteproxyrmse']


## 5. Reconstruction output

The Holocene Reconstruction is saved as a netCDF file.  When the code is done running, the output will be saved in the directory set in 'config.yml' as 'data_dir', under the subdirectory 'results'.  The filename contains the timestamp of when the reconstruction finished. 

### 5.1. Output variables

The variables saved in the output netCDF files are described below.  Terms inside brackets describe the dimensions of the output variables.  Regarding ensemble members, the data assimilation code produces an ensemble of climate states, which is indicated as the 'ens' dimension.  Means and medians of the ensemble members are computed.  For some variables, a randomly selected subset of the ensemble members ('ens_selected') is saved to reduce the file size.

For now, all reconstructed values and proxy inputs are temperature.

#### The gridded Holocene Reconstruction

    recon_mean    [age,lat,lon]               The mean of the Holocene Reconstruction
    recon_median  [age,lat,lon]               The median of the Holocene Reconstruction
    recon_ens     [age,ens_selected,lat,lon]  Selected ensemble members of the Holocene Reconstruction
    recon_gmt     [age,ens]                   The global-mean of the Holocene Reconstruction for all ensemble members
    ages          [age]                       Age (year BP)
    lat           [lat]                       Latitude
    lon           [lon]                       Longitude

#### The proxies

    proxy_values       [proxy,age]       The proxy values, binned and/or nearest neighbor interpolated to the base temporal resolution
    proxy_resolutions  [proxy,age]       The effective temporal resolution of the proxy values, in years
    proxy_uncertainty  [proxy]           The uncertainty values used for each proxy
    proxy_metadata     [proxy,metadata]  Additional metadata about the proxies (datasetname,TSid,lat,lon,seasonality,seasonality_general,median_age_res,collection)

#### Reconstruction of the proxies (at proxy locations and seasonalities)

    proxyrecon_mean    [age,proxy]               The mean of the proxy reconstructions
    proxyrecon_median  [age,proxy]               The median of the proxy reconstructions
    proxyrecon_ens     [age,ens_selected,proxy]  Selected ensemble members of the proxy reconstructions

#### Additional outputs

    options              [options]    The values set in the config.yml file when running the reconstruction
    prior_gmt            [age,ens]    The global-mean of the prior for all ensemble members
    proxies_selected     [proxy]      The proxies which, in theory, were selected for assimilation according to 'percent_to_assimilate'
    proxies_assimilated  [age,proxy]  The proxies which were actually assimilated at each time step.  This may differ from 'proxies_selected' in some cases.

### 5.2. Basic analysis

Within your Holocene Reconstruction directory, the subdirectory 'analysis' contains a Python 3 script for making a simple analysis.  Feel free to use this script as a starting point for more in-depth analysis.

Before running the script, open it in a text editor and update the `recon_dir` and `recon_filename` variables to point to the netCDF output you want to analyze.

---

Should something be added to this readme?  Let me know at michael.erb@nau.edu.
