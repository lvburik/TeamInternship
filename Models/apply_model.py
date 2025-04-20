import numpy as np
import joblib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from thermal_dataset import ThermalDataset 

# load the model
rf = joblib.load("./random_forest_model.joblib")

# choose a test video path
file_path = "fft data/Composite/new_2_lamps_left_off_fft.npy"
file_path = "fft data/Resin/rec_1_lamp_top_on_fft.npy"
file_path = "fft data/Composite/new_2_lamps_right_off_fft.npy"
file_path = "fft data/Resin/rec_2_lamps_right_off_fft.npy"
file_path = "fft data/Resin/circular_3_lamps_fft.npy"
file_paths = [
    "fft data/Composite/new_2_lamps_top_off_fft.npy",
    "fft data/Composite/new_3_lamps_fft.npy",
    "fft data/Resin/circular_1_lamp_left_on_fft.npy",
    "fft data/Resin/circular_1_lamp_right_on_fft.npy",
    "fft data/Resin/circular_1_lamp_top_on_fft.npy",
    "fft data/Resin/circular_2_lamps_left_off_fft.npy",
    "fft data/Resin/circular_3_lamps_fft.npy",
    "fft data/Resin/rec_1_lamp_left_on_fft.npy",
    "fft data/Resin/rec_1_lamp_right_on_fft.npy",
    "fft data/Resin/rec_2_lamps_left_off_fft.npy",
    "fft data/Resin/rec_2_lamps_right_off_fft.npy",
    "fft data/Resin/rec_2_lamps_top_off_fft.npy",
    "fft data/Composite/new_1_lamp_left_on_fft.npy",
    "fft data/Composite/new_1_lamp_right_on_fft.npy",
    "fft data/Composite/new_1_lamp_top_on_fft.npy",
    "fft data/Resin/circular_2_lamps_right_off_fft.npy",
    "fft data/Resin/circular_2_lamps_top_off_fft.npy",
    "fft data/Resin/rec_1_lamp_top_on_fft.npy",
    "fft data/Resin/rec_3_lamps_fft.npy",
    "fft data/Resin/square_2_lamps_left_off_fft.npy",
    "fft data/Resin/square_2_lamps_right_off_fft.npy",
    "fft data/Resin/triangular_1_lamp_right_on_fft.npy",
    "fft data/Resin/triangular_2_lamps_left_off_fft.npy"
]

mask_map = {
        "circular" : "Resin plates/circular.png",
        "rec" : "Resin plates/rec.png",
        "square" : "Resin plates/square.png",
        "tri" : "Resin plates/tri.png",
        "composite" : "Composite plate/composite plate mask.png"
}
data_dir = "/Users/kelseypenners/Library/CloudStorage/OneDrive-TUEindhoven/teaminternship/Experimental Data"


for file_path in file_paths:
    print(f"\nprocessing: {file_path}")

    # load dataset for a video
    dataset = ThermalDataset(
        file_paths=[file_path],
        data_dir=data_dir,
        mask_map=mask_map,
        num_pixels=307200,
        extract_peaks=False,
        cutoff_frequency=0.1
    )

    # get the (fft_data, mask) tuple
    fft_data, mask = dataset[0]

    fft_data = np.abs(fft_data).T

    # predict damaged pixels
    preds = rf.predict(fft_data)

    # reshape to image
    pred_img = preds.reshape((480, 640))

    # visualize predictions
    plt.figure(figsize=(8, 6))
    plt.imshow(pred_img, cmap='hot')
    plt.title(f"predicted defect map\n{file_path}")
    plt.colorbar(label="Class")
    plt.axis("off")
    plt.tight_layout()
   

    # save plot
    file_name = file_path.split('/')[-1].replace('.npy', '.png').replace('_fft', '')
    output_file = f"pred_{file_name}"
    plt.savefig(output_file, dpi=300)
    print(f"saved prediction to {output_file}")

    plt.close()
