# Folder Content 
This folder contains all the code to train and evaluate Random Forest and XGBoost models for defect detection in thermal video data.

# Information per file

## train_model.py
Defines file paths for thermal video and mask data. Used as helper file to decisiontrees.py

## thermal_dataset.py
Contains a custom PyTorch Dataset class for loading '.npy' thermal video data and corresponding masks. Supports preprocessing steps such as FFT, normalization, low-pass filtering, patch extraction, etc. 

**Directories should mirror the following structure:**

Path to Data Dir
  - video files (.npy)
  - masks/

For experimental data, video and mask file paths should include their plate type (composite or resin) and defect shape to ensure thermal_dataset loads the correct correponding mask.


## decisiontrees.py
Main script used to train and test the decision tree models. To use this script, pass in an optional simulation data directory path and an experimental data directory path. 





