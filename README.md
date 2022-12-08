# README FOR HOLOCENE RECONSTRUCTION PROJECT
Written by: Michael P. Erb, Contact: michael.erb@nau.edu

## 1. Introduction

Data and models are two methods of exploring past climate.  Data (such as proxy records) provide point data and models simulate climate changes and climate-system interactions.  The goal of this Holocene Reconstruction project is to use paleoclimate data assimilation--a method of combining information from proxy data and model results--to reconstruct climate over the past 12,000 years.

This GitHub repository contains the Holocene Reconstruction code, and this readme file explains how to set it up and use it.

This code and readme are still under development. To read about the Holocene reconstruction made using v1.0.0 of this code, see Erb et al., in press: "Reconstructing Holocene temperatures in time and space using paleoclimate data assimilation"

## 2. Getting started

The Holocene Reconstruction code is written in Python 3.  The instructions below will help you download the Holocene Reconstruction code, download the necessary data files, install Python 3, and start using the code.

### 2.1. Getting the Holocene Reconstruction code

To get v1.0.0 of the code, search for "Holocene reconstruction code, v1.0.0" on Zenodo.

Alternately, you may be able to find a newer version of code on Github. Clone the Github repository into your Linux environment with the command:

    git clone https://github.com/Holocene-Reconstruction/Holocene-code.git

### 2.2. Getting the necessary data files

The code uses climate model output as well as proxy data files.  To get this data, download the zip file from this link: https://doi.org/10.5281/zenodo.7407116

Put this file in a convenient place and unzip it using `unzip holocene_da_data.zip`.  It should contain the following subdirectories:

    models/   Model output
    proxies/  Proxy files
    results/  Results of the data assimilation (initially empty)

The data set includes TraCE-21ka temperature output, HadCM3 temperature output, and Temp12k proxies.

### 2.3. Installing Python 3 and necessary packages

Make sure that you have Python 3 installed.  If you don't, one option is the package manager Anaconda: [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)

Most of the necessary packages should come with the standard Python 3 installation, installed using `conda create -n python3_da anaconda`.  The only ones you should need to install are xarray, netCDF4, and lipd.  Either install these yourself (note: LiPD can be installed with `pip install LiPD`) or go to your Holocene Reconstruction directory and use the commands below to create a new Python 3 environment with the necessary packages and switch to it:

    conda env create -f environment_da.py
    conda activate python3_da

### 2.4. First-time setup

Before running the Holocene Reconstruction code for the first time, do the following:
 1. Open config_default.py.
 2. Near the top, change the 'data_dir' directory to the location of the unzipped data from section 2.2 above (i.e., the directory which has the models/, proxies/, and results/ subdirectories.)
 3. Save config_default.py and copy it to config.py.  You can set up new experiments in config.py while keeping the original file for reference.

The "default" settings are used for the main experiment shown in Erb et al., in press "Reconstructing Holocene temperatures in time and space using paleoclimate data assimilation".

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

By default, the Holocene Reconstruction code reconstructs the past 12,000 years.  This is set in the `age_range_to_reconstruct` variable (default: [0,12000]), but can be shortened or lengthened as requested.  The Temp12k proxy database contains proxy data older than 12ka, but the quality of the reconstruction will decline as the number of available proxies diminish further back in time.  Another relevant variable is `reference_period` (default: [3000,5000]), which defines the reference period used for each proxy.  When doing a reconstruction relative to a reference period, proxies without data in the selected reference period will not be used.

#### Proxies to assimilate

The variable `proxy_datasets_to_assimilate` (default: ['temp12k']) lists the proxy database(s) to be assimilated.  Only Temp12k proxies are supported out-of-the-box.  Other proxy datasets, such as PAGES2k, may be added later.

#### Models to use as the prior

The variable `models_for_prior` (default: ['hadcm3_regrid','trace_regrid']) defines the model simulation(s) to be used as the model prior.

#### The size of the prior 'window'

The model prior consists of a set of model states chosen from the simulation(s) specified above.  By default, the code doesn't use all modeled climate states, but selects a 'window' of time surrounding each year of the reconstruction.  The length of this window, in years, is set in the variable `prior_window` (default: 5010).  A smaller value will allow the prior to change more through time, but too few model states may result in a poorer reconstruction, and vice versa.  To use all climate states from the model prior (resulting in a prior that does not change with time) set this to 'all'.

#### Time resolutions

The Holocene Reconstruction uses a multi-timescale approach to data assimilation, which can assimilate proxies at a variety of timescales.  To change the details of how this works, the variables `time_resolution` (default: 10) and `maximum_resolution` (default: 1000) can be changed.  `time_resolution` defines the base temporal resolution, in years, used in the reconstruction.  Proxy data will be preprocessed at this resolution and the final reconstruction will be output at this resolution.  Few proxies in the Temp12k database have temporal resolution finer than 10 years, so little information should be lost when using this base resolution.  Changing this number should also affect the speed of the data assimilation, but this has not been well-tested.  `maximum_resolution` specifies the maximum temporal resolution, in years, that proxy data points will be treated as.  This option does not affect which proxies are assimilated or omitted, but only affects the timescale which is used when translating proxy data to the larger climate system; proxy data points which appear to represent longer periods of time will be assumed to represent this amount of time instead.

#### Localization radius

If a localization radius is desired, set it in the `localization_radius` (default: None) variable, in units of meters.  To use a localization radius, the variable `assimate_together` (default: True) should be set to False, or else the localization radius will not be used.  Setting `assimilate_together` to False will probably slow down the code, since it will assimilate proxies one at a time instead of all together.  However, this must be done for a localization radius to work in this code.

#### Assimilating only a portion of the proxy database

To only assimilate some of the proxies in the database, modify the variables `assimilate_selected_seasons`, `assimilate_selected_archives`, `assimilate_selected_region`, and/or `assimilate_selected_resolution`. See the config file for more details.

Additionally, to assimilate only a portion of the proxy database, set the variable `percent_to_assimilate` (default: 100) to a lower number.

#### Using pseudoproxies

To generate/use pseudoproxies, the string in 'proxy_datasets_to_assimilate' should be given in the form ['pseudo_VAR1_using_VAR2_noise_VAR3'], where:

    VAR1: The space/time structure of the pseudoproxy database (e.g. 'temp12k','basicgrid10','basicgrid5') 
    VAR2: The model that the pseudoproxies are generated from (e.g. 'hadcm3','trace','famous')
    VAR3: The noise added to the pseudoproxies (e.g. 'none','whiteproxyrmse')

Example: ['pseudo_temp12k_using_hadcm3_noise_whiteproxyrmse']


## 5. Reconstruction output

The Holocene Reconstruction is saved as a netCDF file.  When the code is done running, the output will be saved in the directory set in 'config.yml' as 'data_dir', under the subdirectory 'results'.  The filename contains the timestamp of when the reconstruction finished. 

### 5.1. Output variables

The variables saved in the output netCDF files are described below.  Terms inside brackets describe the dimensions of the output variables.  Regarding ensemble members, the data assimilation code produces an ensemble of climate states, which is indicated as the 'ens' dimension.  Means and medians of the ensemble members are computed.  For some variables, a randomly selected subset of the ensemble members ('ens_selected') is saved to reduce the file size.

For now, all reconstructed values and proxy inputs are temperature.

#### The gridded Holocene Reconstruction

    recon_tas_mean        [age,lat,lon]               The ensemble-mean of the Holocene Reconstruction
    recon_tas_ens         [age,ens_selected,lat,lon]  Selected ensemble members of the Holocene Reconstruction
    recon_tas_global_mean [age,ens]                   The global-mean of the Holocene Reconstruction for all ensemble members
    recon_tas_nh_mean     [age,ens]                   The NH-mean of the Holocene Reconstruction for all ensemble members
    recon_tas_sh_mean     [age,ens]                   The SH-mean of the Holocene Reconstruction for all ensemble members
    ages                  [age]                       Age (year BP)
    lat                   [lat]                       Latitude
    lon                   [lon]                       Longitude

#### The prior

    prior_tas_mean        [age,lat,lon]  The ensemble-mean of the prior
    prior_tas_global_mean [age,ens]      The global-mean of the prior for all ensemble members

#### The proxies

    proxy_values       [age,proxy]       The proxy values, binned and/or nearest neighbor interpolated to the base temporal resolution
    proxy_resolutions  [age,proxy]       The effective temporal resolution of the proxy values, in years
    proxy_uncertainty  [proxy]           The uncertainty values used for each proxy
    proxy_metadata     [proxy,metadata]  Additional metadata about the proxies (datasetname,TSid,lat,lon,seasonality,seasonality_general,median_age_res_calc*,collection)  *The "median_age_res_calc" variable may not accurately represent the proxy resolution. You may want to load the proxy data and calculate this yourself.

#### Reconstruction of the proxies (at proxy locations and seasonalities)

    proxyrecon_mean    [age,proxy]               The mean of the proxy reconstructions
    proxyrecon_ens     [age,ens_selected,proxy]  Selected ensemble members of the proxy reconstructions

#### Additional outputs

    options              [options]    The values set in the config.yml file when running the reconstruction
    proxies_selected     [proxy]      The proxies which were selected for assimilation
    proxies_assimilated  [age,proxy]  The proxies which were actually assimilated at each time step.  This may differ from 'proxies_selected' in some cases.
    proxyprior_mean      [age,proxy]  The mean of the prior estimates of the proxies

---

Should something be added to this readme?  Let me know at michael.erb@nau.edu.
