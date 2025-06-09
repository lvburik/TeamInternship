# Folder Content 
This folder contains all code to train and evaluate the implemented CNN and UNets discussed in the report.

# Information per file

## train_model.py
Defines file paths for thermal video and mask data. Used as helper file to nns.py

## thermal_dataset.py
Contain a custom PyTorch Dataset class for loading '.npy' thermal video data and corresponding masks. Performs option preprocessing steps such as FFT, normalization, low-pass filtering, patch extraction, etc. 

## decisiontrees.py
Main script used to train and test the CNN and UNet models.