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

Put this fine in a convenient place and it using `unzip holocene_da_data.zip`.  It should contain the following subdirectories:

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
 2. Go to your Holocene Reconstruction directory and open config_default.yml.  Change the "lipd_dir" line near the top to specify the location of the newly-created lipd directory.  If you're not using this workaround, feel free to delete the 'lipd_dir' line in config_default.yml.

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

### 3.1. Experimental options

Running the code as-is will run the default experiment.  To change options and set up new experiments, change variables in the "config.py" file.  These variables are explained below.

[Add explanation of variables.]

