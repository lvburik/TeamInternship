# Folder Content

This folder contains code for running inference using a trained SegFormer model on thermal video data. Four .py files are present - two for the training and evaluation of the SegFormer for raw data and two for the training and evaluation for the TSR data. One more .py file is used for training, based on all videos. Furthermore, this folder containes images of predicted masks for specific frames and different shapes with both confidence interval and with thresholding. Last, but not least, this folder includes trained model weights .pth files for both the raw data and the TSR data. 

## evaluate_segformer.py
Full evaluation script for a SegFormer model, based on raw data, visualization of predicted masks. 
## evaluate_segformer_tsr.py
Full evaluation script for a SegFormer model, based on TSR data, visualization of predicted masks. 
## segformer.py
Full training script for a SegFormer model, based on raw data, taking into account all videos.
## train_segformer_on_six.py
Full training script for a SegFormer model, based on raw data, taking into account a subset of videos.
## train_segformer_tsr.py
Full training script for a SegFormer model, based on TSR data, taking into account a subset of videos.
## segformer_tsr_25ep.pth
File, containing the weights for a SegFormer model, trained on TSR data with all videos and 25 epochs
## segformer_weights_for_15_epochs_all.pth
File, containing the weights for a SegFormer model, trained on raw data with 6 videos and 15 epochs
## segformer_weights_for_tsr_15_epochs_all.pth
File, containing the weights for a SegFormer model, trained on TSR data with 6 videos and 15 epochs
## segformer_50ep.pth
File, containing the weights for a SegFormer model, trained on raw data with all videos and 50 epochs
## prediction_rec.png
Predicted mask for frame in experimental video for a rectangular shaped defect with thresholding. 
## prediction_rec_softmax.png
Predicted mask for frame in experimental video for a rectangular shaped defect with confidence score. 
## prediction_square.png
Predicted mask for frame in experimental video for a square shaped defect with thresholding. 
## prediction_triangle.png
Predicted mask for frame in experimental video for a triangular shaped defect with thresholding. 
## triangle_prediction_softmax.png
Predicted mask for frame in experimental video for a triangular shaped defect with confidence score. 
## prediection_circular.png
Predicted mask for frame in experimental video for a circular shaped defect with thresholding. 
## segformer_softmax.png
Predicted mask for three frames in experimental video for a circular shaped defect with confidence scores. 
## segformer_threshold.png
Predicted mask for three frames in experimental video for a circular shaped defect with thresholding. 




