import os
import sys
import torch
from torch.utils.data import DataLoader
import joblib
import torchmetrics
from sklearn.preprocessing import StandardScaler
from thermal_dataset import ThermalDataset
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# define experimental training data set
TRAIN_DATA = [
    "fft data/Composite/new_2_lamps_left_off_fft.npy",
    "fft data/Composite/new_2_lamps_right_off_fft.npy",
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
    "fft data/Resin/square_1_lamp_left_on_fft.npy",
    "fft data/Resin/square_1_lamp_right_on_fft.npy",
    "fft data/Resin/square_1_lamp_top_on_fft.npy",                   
    "fft data/Resin/square_2_lamps_top_off_fft.npy",
    "fft data/Resin/square_3_lamps_fft.npy",
    "fft data/Resin/triangular_1_lamp_left_on_fft.npy",              
    "fft data/Resin/triangular_1_lamp_top_on_fft.npy",            
    "fft data/Resin/triangular_2_lamps_right_off_fft.npy",
    "fft data/Resin/triangular_2_lamps_top_off_fft.npy",
    "fft data/Resin/triangular_3_lamps_fft.npy"]

# define experimental test data set
TEST_DATA = [
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
    "fft data/Resin/triangular_2_lamps_left_off_fft.npy"] 

# define experimental labels
MASK_MAP = {
        "circular" : "Resin plates/circular.png",
        "rec" : "Resin plates/rec.png",
        "square" : "Resin plates/square.png",
        "tri" : "Resin plates/tri.png",
        "composite" : "Composite plate/composite plate mask.png"
}

NUM_PIXELS = 307200

def calculate_iou(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # calculate intersection over union
    iou = tp / (tp + fn + fp) if tp + fn + fp > 0 else 0

    return iou

def evaluate(y_true, y_pred):

    # calculate evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    iou = calculate_iou(y_true, y_pred)

    return acc, precision, recall, f1, iou


def main(sim_data_path, exp_data_path):
    
    # initialize training dataset
    train_dataset = ThermalDataset(
        file_paths=TRAIN_DATA,
        data_dir=exp_data_path, 
        mask_map=MASK_MAP, 
        num_pixels=NUM_PIXELS,
        apply_PCA=False,
        extract_peaks=False,
        extract_patches=True,
        cutoff_frequency=0.1
    )
    
    # initialize testing dataset
    test_dataset = ThermalDataset(
        file_paths=TEST_DATA,
        data_dir=exp_data_path,
        mask_map=MASK_MAP,
        num_pixels=NUM_PIXELS, 
        extract_peaks=False,
        extract_patches=True,
        cutoff_frequency=0.1
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    X_train, y_train = [], []
    X_test, y_test = [], []

    #freq = np.load(os.path.join(exp_data_path, "exp_freq.npy"))

    # train data
    for batch_data in train_dataloader:
        fft_data, mask = batch_data

        # format data
        fft_data = np.abs(fft_data).squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()
        
        """# randomly select pixels
        pixel_id = np.random.choice(NUM_PIXELS, 10)

        # plot randomly selected pixels
        p_fft = filtered_fft[:, pixel_id]
        plt.figure(figsize=(8, 4))
        plt.plot(freq[:700], np.abs(p_fft[:700]))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'FFT of Thermal Signal (Filtered) for Pixel {pixel_id}')
        plt.grid(True)
        plt.show()"""
    
        # turn into (num_pixels, num_freqs)
        fft_data = fft_data.T

        # turn into (num_pixels, )
        mask = mask.flatten()

        X_train.append(fft_data)
        y_train.append(mask)
    
    # put train data into one array
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    #x, y = resample(X_train, y_train, n_samples=100000, random_state=42)

    print("training random forest")
    
    # train random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # check batch of data
    for batch_data in test_dataloader:
        fft_data, mask = batch_data
        
        # req = freq.squeeze(0)
        fft_data = np.abs(fft_data).squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()

        # turn into (num_pixels, num_frames)
        fft_data = fft_data.T

        # turn into (num_pixels, )
        mask = mask.flatten()

        X_test.append(fft_data)
        y_test.append(mask)
    
    # put test data into one array
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    #x, y = resample(X_test, y_test, n_samples=80000, random_state=42)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # predict on test data
    preds = rf.predict(X_test)

    # calculate evaluation metrics
    acc, precision, recall, f1, iou = evaluate(y_test, preds)
    print("Random Forest Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")
    
    joblib.dump(rf, "rf_batch.joblib")

    # random forest accuracy run on fft data with 100k samples
    #   0.904

    # Random Forest Evaluation: trained with 2 exp 100k samples
    #   Accuracy: 0.5793
    #   Precision: 0.9422
    #   Recall: 0.5746
    #   F1 Score: 0.7139
    #   IoU: 0.5550

    # Random Forest Evaluation: trained with 10k pixels
    #   Accuracy: 0.9105
    #   Precision: 0.9258
    #   Recall: 0.9809
    #   F1 Score: 0.9525
    #   IoU: 0.9094

    # Random Forest Evaluation: trained with 300k pixels
    #   Accuracy: 0.9056
    #   Precision: 0.9323
    #   Recall: 0.9667
    #   F1 Score: 0.9492
    #   IoU: 0.9033

    # Random Forest Evaluation: trained with 900k pixels
    #   Accuracy: 0.8994
    #   Precision: 0.9342
    #   Recall: 0.9573
    #   F1 Score: 0.9456
    #   IoU: 0.8969

    # Random Forest Evaluation: trained on patches of 4x4
    #   Accuracy: 0.8602
    #   Precision: 0.9326
    #   Recall: 0.9123
    #   F1 Score: 0.9223
    #   IoU: 0.8559

if __name__ == "__main__":
    # read file path for data
    sim_data_path = sys.argv[1]
    exp_data_path = sys.argv[2]
    main(sim_data_path, exp_data_path)
