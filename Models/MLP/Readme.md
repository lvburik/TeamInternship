# Folder Content 
This folder contains all code to train and evaluate a MLP as discussed in the report

# Information per file

## Dataset_to_HDF5.py
converts data and masks from individual .npy, and .png files to a single .h5 dataset. This is done to improve performance during training using the dataloader.

## Linear_model.py
file used to train and evaluate the MLP models. can be run using Vscodes interactive python, or all at once. saves a trained network

## Training_and_evaluation.py
contains all functions used for training and evaluation of the linear model

## models_dataset_dataloader.py
contains the code for the dataset, augmentation and dataloader used in Linear_model.py

