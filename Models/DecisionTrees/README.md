# Folder Content 
This folder contains all code to train and evaluate Random Forest and XGBoost models for defect detection in thermal video data.

# Information per file

## train_model.py
Defines file paths for thermal video and mask data. Used as helper file to decisiontrees.py

## thermal_dataset.py
Contain a custom PyTorch Dataset class for loading '.npy' thermal video data and corresponding masks. Performs option preprocessing steps such as FFT, normalization, low-pass filtering, patch extraction, etc. 

## decisiontrees.py
Main script used to trian and test the decision tree models


