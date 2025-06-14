import torch
import joblib
import torch.nn as nn
import matplotlib.pyplot as plt
from Models.CNNandUNets.thermal_dataset import ThermalDataset 
from Models.CNNandUNets.nns import *
from Models.CNNandUNets.train_model import *

# choose 'rf' or 'cnn' or 'unet' or 'smallunet'
apply_model = 'cnn'

mask_map = {
        "circular" : "Resin plates/circular.png",
        "rec" : "Resin plates/rec.png",
        "square" : "Resin plates/square.png",
        "tri" : "Resin plates/tri.png",
        "composite" : "Composite plate/composite plate mask.png"
}

# define path to data
data_dir = "/Users/kelseypenners/Library/CloudStorage/OneDrive-TUEindhoven/teaminternship/Experimental Data"

# reconstruction of predicted frame from patch predictions
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

for file_path in TEST_DATA:
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
    
    # apply rf model
    # rf_sampledpixels.joblib or rf_patches.joblib
    if apply_model == 'rf':
        rf = joblib.load("./Models/saved models/rf_sampledpixels.joblib")
        preds = rf.predict(fft_data.T)
        pred_img = preds.reshape((480 , 640))

    # apply neural network models
    # depending on input data, change number of in channels!
    if apply_model == 'cnn' or apply_model == 'unet' or apply_model == 'smallunet':
        if apply_model == 'unet':
            model = UNet(in_channels=10)
        if apply_model == 'cnn':
            model = Network(n_in=103)
        if apply_model == 'smallunet':
            model = SmallerUNet(in_channels=10)
        model.load_state_dict(torch.load("./Models/saved models/cnn_size64patches_400_nopca.pth", map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            # make predictions
            fft_data = torch.tensor(fft_data).permute(0, 3, 1, 2).float()
            preds = model(fft_data)
            preds = torch.sigmoid(preds)

            # tensor of shape [patches, 1, 128, 128]
            preds = preds.cpu().detach().squeeze(1)

            # normalize patch predictions
            def normalize(patch):
                min_val = patch.min()
                max_val = patch.max()
                if max_val - min_val > 0:
                    return (patch - min_val) / (max_val - min_val)
                else:
                    return patch * 0  # completely flat patch
                
            norm_preds = torch.stack([normalize(p) for p in preds])
            
        # reconstruct the patches to the original image size
        # adjust patch_size and overlap to match thermal_dataset patch extraction settings
        reconstructed = reconstruct_patches(preds, patch_size=64, overlap=0.75)

        pred_img = reconstructed.cpu().numpy()
    
    # for thresholding
    #pred_img = (pred_img > 0.5).astype(np.uint8) 

    # visualize predictions
    plt.figure(figsize=(8, 6))
    plt.imshow(pred_img, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label="confidence")
    plt.axis("off")
    plt.tight_layout()
   
    # save prediction plot
    file_name = file_path.split('/')[-1].replace('.npy', '.png').replace('_fft', '')
    output_file = f"pred_{file_name}"
    plt.savefig(output_file, dpi=300)
    print(f"saved prediction to {output_file}")

    plt.close()
