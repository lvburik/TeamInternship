import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from thermal_dataset import ThermalDataset 
from nns import *
import math
from train_model import evaluate

# choose 'rf' or 'xgb' or 'cnn' or 'unet' or 'smallunet'
apply_model = 'smallunet'

file_paths = [
    # training data
    "fft data/Resin/circular_1_lamp_left_on_fft.npy",
    "fft data/Resin/circular_1_lamp_right_on_fft.npy",
    "fft data/Resin/circular_1_lamp_top_on_fft.npy",
    "fft data/Resin/circular_2_lamps_left_off_fft.npy",
    "fft data/Resin/circular_3_lamps_fft.npy",
    "fft data/Resin/rec_2_lamps_right_off_fft.npy",
    "fft data/Resin/rec_2_lamps_top_off_fft.npy",                
    "fft data/Resin/square_1_lamp_left_on_fft.npy",
    "fft data/Resin/square_1_lamp_right_on_fft.npy",
    # test data
    "fft data/Composite/new_1_lamp_left_on_fft.npy",
    "fft data/Composite/new_1_lamp_right_on_fft.npy",
    "fft data/Composite/new_1_lamp_top_on_fft.npy",
    "fft data/Resin/circular_2_lamps_right_off_fft.npy",
    "fft data/Resin/circular_2_lamps_top_off_fft.npy",
    "fft data/Resin/square_2_lamps_left_off_fft.npy",
    "fft data/Resin/square_2_lamps_right_off_fft.npy",
    "fft data/Resin/triangular_1_lamp_right_on_fft.npy",]

# train
"""fft data/Resin/rec_1_lamp_left_on_fft.npy",
    "fft data/Composite/new_2_lamps_left_off_fft.npy",
    "fft data/Composite/new_2_lamps_right_off_fft.npy",
    "fft data/Composite/new_2_lamps_top_off_fft.npy",
    "fft data/Composite/new_3_lamps_fft.npy",
    "fft data/Resin/rec_1_lamp_right_on_fft.npy",
    "fft data/Resin/rec_2_lamps_left_off_fft.npy",
    "fft data/Resin/square_1_lamp_top_on_fft.npy",                   
    "fft data/Resin/square_2_lamps_top_off_fft.npy",
    "fft data/Resin/square_3_lamps_fft.npy",
    "fft data/Resin/triangular_1_lamp_left_on_fft.npy",              
    "fft data/Resin/triangular_1_lamp_top_on_fft.npy",            
    "fft data/Resin/triangular_2_lamps_right_off_fft.npy",
    "fft data/Resin/triangular_2_lamps_top_off_fft.npy",
    "fft data/Resin/triangular_3_lamps_fft.npy"""
# test
"""fft data/Resin/rec_1_lamp_top_on_fft.npy",
    "fft data/Resin/rec_3_lamps_fft.npy",
    "fft data/Resin/triangular_2_lamps_left_off_fft.npy"""

file_paths = [f.replace('fft data', 'fftpca data') for f in file_paths]
file_paths = [f.replace('_fft', '_fftpca') for f in file_paths]
mask_map = {
        "circular" : "Resin plates/circular.png",
        "rec" : "Resin plates/rec.png",
        "square" : "Resin plates/square.png",
        "tri" : "Resin plates/tri.png",
        "composite" : "Composite plate/composite plate mask.png"
}
data_dir = "/Users/kelseypenners/Library/CloudStorage/OneDrive-TUEindhoven/teaminternship/Experimental Data"

def reconstruct_patches(pred_patches, patch_size=128, overlap=0):
    image_size = (480, 640)
    stride = int(patch_size * (1 - overlap))
    H, W = image_size

    # patch positions
    grid_i = list(range(0, H - patch_size + 1, stride))
    grid_j = list(range(0, W - patch_size + 1, stride))
    n_grid_patches = len(grid_i) * len(grid_j)

    # regular patches and irregular edge patches
    main_grid_patches = pred_patches[:n_grid_patches]
    extra_patches = pred_patches[n_grid_patches:]

    patches_flat = main_grid_patches.view(main_grid_patches.shape[0], -1).T.unsqueeze(0)
    fold = nn.Fold(
        output_size=image_size,
        kernel_size=patch_size,
        stride=stride
    )
    output = fold(patches_flat)
    ones = torch.ones_like(patches_flat)
    divisor = fold(ones)
    divisor[divisor == 0] = 1e-6
    result = (output / divisor).squeeze(0).squeeze(0)

    # manually add extra patches (bottom, right, bottom-right)
    idx = 0

    # bottom edge
    if (H - patch_size) % stride != 0:
        i = H - patch_size
        for j in grid_j:
            patch = extra_patches[idx]
            result[i:i+patch_size, j:j+patch_size] = patch
            idx += 1
    # right edge
    if (W - patch_size) % stride != 0:
        j = W - patch_size
        for i in grid_i:
            patch = extra_patches[idx]
            result[i:i+patch_size, j:j+patch_size] = patch
            idx += 1
    # bottom right corner
    if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
        i, j = H - patch_size, W - patch_size
        patch = extra_patches[idx]
        result[i:i+patch_size, j:j+patch_size] = patch

    return result

for file_path in file_paths:
    print(f"\nprocessing: {file_path}")

    # load dataset for a video
    dataset = ThermalDataset(
        file_paths=[file_path],
        data_dir=data_dir,
        mask_map=mask_map,
        extract_patches=False,
        extract_cnn_patches = (apply_model == 'cnn' or apply_model == 'unet' or apply_model == 'smallunet'),
    )

    # get the (fft_data, mask) tuple
    fft_data, mask = dataset[0]

    print(f"fft data shape: {fft_data.shape}")

    # apply xgb model
    if apply_model == 'xgb':
        xgb = joblib.load("./Models/saved models/xgb.joblib")
        preds = xgb.predict(fft_data.T)
        pred_img = preds.reshape((480 , 640))
    
    # apply rf model
    if apply_model == 'rf':
        rf = joblib.load("./Models/saved models/rf_batch_rn.joblib")
        preds = rf.predict(fft_data)
        pred_img = preds.reshape((480 , 640))

    if apply_model == 'cnn' or apply_model == 'unet' or apply_model == 'smallunet':
        if apply_model == 'unet':
            model = UNet(in_channels=10)
        if apply_model == 'cnn':
            model = Network(n_in=10)
        if apply_model == 'smallunet':
            model = SmallerUNet(in_channels=10)
        model.load_state_dict(torch.load("./Models/saved models/smallunet_pca_200.pth", map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            # make predictions
            fft_data = torch.tensor(fft_data).permute(0, 3, 1, 2).float()
            print(f"fft data shape: {fft_data.shape}")
            preds = model(fft_data)
            preds = torch.sigmoid(preds)

            print(f"preds shape: {preds.shape}")
            print(f"Raw preds stats for {file_path}: min={preds.min().item()}, max={preds.max().item()}, mean={preds.mean().item()}")
            #preds = preds.view(-1).cpu().numpy().astype(int)

            # tensor of shape [patches, 1, 128, 128]
            preds = preds.cpu().detach().squeeze(1)

            def normalize(patch):
                min_val = patch.min()
                max_val = patch.max()
                if max_val - min_val > 0:
                    return (patch - min_val) / (max_val - min_val)
                else:
                    return patch * 0  # completely flat patch
                
            norm_preds = torch.stack([normalize(p) for p in preds])
            
        # reconstruct the patches to the original image size
        reconstructed = reconstruct_patches(preds, patch_size=32, overlap=0.5)
        
        #predicted = (preds > 0.5).float()
        #preds = predicted.view(-1).cpu().numpy().astype(int)
        # reshape to image
        #pred_img = preds.reshape((480 , 640))

        pred_img = reconstructed.cpu().numpy()
    
    #pred_img = (pred_img > 0.5).astype(np.uint8) 
    # visualize predictions
    plt.figure(figsize=(8, 6))
    plt.imshow(pred_img, cmap='gray', vmin=0, vmax=1)
    plt.title(f"predicted defect map")
    plt.colorbar(label="confidence (0=nondefect, 1=defect)")
    plt.axis("off")
    plt.tight_layout()
   
    # save plot
    file_name = file_path.split('/')[-1].replace('.npy', '.png').replace('_fft', '')
    output_file = f"pred_{file_name}"
    plt.savefig(output_file, dpi=300)
    print(f"saved prediction to {output_file}")

    plt.close()
