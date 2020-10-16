# README FOR HOLOCENE RECONSTRUCTION PROJECT
Written by: Michael P. Erb, Contact: michael.erb@nau.edu

## 1. Introduction

Data and models are two methods of exploring past climate, with data (such as proxy records) providing point data and models simulating climate changes and climate-system interactions.  The goal of this Holocene Reconstruction project is to use data assimilation, a method of combining information from proxy data and model results, to reconstruct climate over the past 12,000 years.

This GitHub repository contains the Holocene Reconstruction code, and this readme file explains how to set up and run this code.

This code and readme are still under development.

## 2. Getting started

The Holocene Reconstruction code is written in Python 3.  The instructions below will help you download the Holocene Reconstruction code, download the necessary data files, install Python 3, and start using the code.

### 2.1. Getting the Holocene Reconstruction code

Clone this code into your Linux environment with the command:

    git clone https://github.com/Holocene-Reconstruction/Holocene-code.git

### 2.2. Getting the necessary data files

The code uses climate model output as well as proxy data files.  To get this data, download the zip file from this link: [data on Google Drive](https://drive.google.com/file/d/1Iqfbpa4mhoIw_ccKYzfljkkTTz47HKzJ/view?usp=sharing)

Put this fine in a convenient place and unzip it using `unzip holocene_da_data.zip`.  It should contain the following subdirectories:

    models/   Model output
    proxies/  Proxy files
    results/  Results of the data assimilation (initially empty)

The data currently includes only the TraCE model output and Temp12k proxies.  Other data may be added later.

### 2.3. Installing Python 3 and necessary packages

Make sure that you have Python 3 installed.  If you don't already have it, one option is the package manager Anaconda: [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)

Most of the necessary packages should come with the standard Python 3 installation.  The only ones you may need to install are "pyyaml" and "lipd".  Either install these yourself (note: LiPD can be installed with `pip install LiPD`) or go to your Holocene Reconstruction directory and use the commands below to create a new Python 3 environment with the necessary packages and switch to it:

    conda env create -f environment_da.py
    conda activate python3_da

NOTE: If you have trouble installing the LiPD library, you can use the following workaround:
 1. Clone the github repository at [https://github.com/nickmckay/LiPD-utilities](https://github.com/nickmckay/LiPD-utilities) to a convenient location
 2. Go to your Holocene Reconstruction directory and open config_default.yml.  Change the "lipd_dir" line near the top to specify the location of the newly created LiPD directory.  If you're not using this workaround, feel free to delete the 'lipd_dir' line in config_default.yml.

### 2.4. First-time setup

Before running the Holocene Reconstruction code for the first time, several things must be done:
 1. Open config_default.py.
 2. Near the top, change the 'data_dir' directory to point to the location where you unzipped the data file in section 2.2 above.
 3. Save config_default.py and copy it to config.py.  You can use config.py to make changes, keeping the default file unaffected.

## 3. Running the code

You should now be ready to run the data assimilation code.  To run the code using the default options, go to your Holocene Reconstruction directory and execute the command:

    python da_main_code.py config.yml

The code will update you on what it's doing as it runs.  However, since the code can take some time to run, you may want to submit it in the background or use a job scheduler.  If you use Slurm, you can use the run_da.sh file.  Double-check the settings in that file and run it with:

    sbatch run_da.sh

## 4. Setting up new experiments

Running the code as-is will run the default experiment.  To change options and set up new experiments, change variables in the `config.yml` file.

### 4.1. Experimental options

NOTE: THIS SECTION IS STILL BEING WRITTEN.  FORGIVE ANY INCOMPLETE OR CONFUSING EXPLANATIONS.

Some of the variables that you can set in `config.yml` are explained below.  Additional variables can be seen by opening the config.yml file.  Some of these options have not be thoroughly tested, so some experimental designs may result in crashing the code.

#### Age range to reconstruct

By default, the Holocene Reconstruction code reconstructs the past 12,000 years.  This is set in the `age_range_to_reconstruct` variable (default: [0,12000]), but can be shortened or lengthened as requested.  The Temp12k proxy database contains some proxy data older than 12ka, but the quality of the reconstruction will decline as the number of available proxies diminish.  Another relevant variable is `reference_period` (default: [0,5000]), which defines the reference period used for every proxy.  Proxies without data in this reference period will not be used.

#### Proxies to assimilate

The variable `proxy_datasets_to_assimilate` (default: ['temp12k']) lists the proxy database(s) that should be assimilated.  For now, only Temp12k proxies are supported out-of-the-box.  Other proxy datasets, such as PAGES2k, may be added later.

#### Models to use as the prior

The variable `models_for_prior` (default: ['trace_regrid']) defines the model simulation(s) which will be used as the model prior.  Only TraCE is included in the download in section 2.2 above, but I'll add HadCM3 soon as well.

#### The size of the prior 'window'

The model prior consists of a set of model states chosen from the simulation specified above.  By default, the code does not use all of the modeled climate states, but selects a 'window' of time surrounding each year of the reconstruction.  The length of this window, in years, is set in the variable `prior_window` (default: 5010).  A smaller value will allow the prior to change more through time, but too few model years may result in a degraded reconstruction, and vice versa.  To use all climate states from the model prior (resulting in a prior that does not change with time) set this to 'all'.

#### Time resolutions

The Holocene Reconstruction uses a multi-timescale approach to data assimilation, which can assimilate proxies at a variety of timescales.  To set exactly how this works, the variables `time_resolution` (default: 10) and `maximum_resolution` (default: 1000) can be changed.  `time_resolution` defines the base temporal resolution that the code will use, in years.  Proxy data will be processed at this resolution, and the final reconstruction will be output at this resolution.  Few proxies in the Temp12k database have temporal resolution finer than this.  Altering this number should also affect the speed of the data assimilation, but this has not been well-tested.  `maximum_resolution` specifies the maximum temporal resolution for processing proxies, in years.  This option is a bit technical, and does not affect which proxies are assimilated or not.  Instead, it only affects the timescale which is used when translating proxy data to the larger climate system; proxy data points which appear to represent longer periods of time will be assumed to represent this amount of time instead.  [TODO: improve this explanation.]

#### Localization radius

If a localization radius is desired, set it in the `localization_radius` (default: None) variable.  The localization radius should be in meters.  For this to work, the variable `assimate_together` (default: True) should be set to False.  Doing this will significantly slow down the data assimilation, since assimilating proxies one at a time is more time-consuming, but must be done for a localization radius to work.

#### Assimilating only a portion of the proxy database

By default, all valid proxies are assimilated.  To change this, set the variable `percent_to_assimilate` (default: 100) to a lower number.

## 5. Reconstruction output

The Holocene Reconstruction is saved as a netCDF file.  When the code is done running, the output will be saved in the directory set in 'config.yml' as 'data_dir', under the subdirectory 'results'.  The filename contains the timestamp of when the reconstruction finished. 

### 5.1. Output variables

The variables saved in the output netCDF files are as follows.  Values inside brackets indicates the dimensions of the output variables.  The data assimilation code produces an ensemble of climate states, which is indicated as the 'ens' dimension.  For some variables, a randomly selected subset of the ensemble members is saved to reduce the file size.

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

    options              [options]    Values of options set in the config.yml file when running the reconstruction
    prior_gmt            [age,ens]    The global-mean of the prior for all ensemble members
    proxies_selected     [proxy]      The proxies which, in theory, were selected for assimilation according to 'percent_to_assimilate'
    proxies_assimilated  [age,proxy]  The proxies which were actually assimilated at each time step.  This may differ from 'proxies_selected' in some cases.

### 5.2. Basic analysis

Within your Holocene Reconstruction directory, the subdirectory 'analysis' contains a Python 3 script for making a simple analysis.

Before running this script, open the script in a text editor and update the `recon_dir` and `recon_filename` variables to point to your reconstruction netCDF output.

Feel free to use this script as a starting point for more in-depth analysis.


Should something be added to this readme?  Let me know at michael.erb@nau.edu.
