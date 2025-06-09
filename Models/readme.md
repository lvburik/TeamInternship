# Folder Content

This folder contains code for running inference using a trained SegFormer model on thermal video data. Under each of the model folders, training and evaluation code can be found with corresponding readmes.

## segformer_inference.py
Main script for loading a saved SegFormer model and running inference on a thermal video ('.npy' format). Outputs and saves a defect confidence map.

Run with:
python segformer_inference.py path/to/video.npy

