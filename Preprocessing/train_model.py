import os
import sys
import torch
from torch.utils.data import DataLoader
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from thermal_dataset import ThermalDataset
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


TRAIN_DATA =[
    "Composite plate/new_2_lamps_left_off.npy",
    "Composite plate/new_2_lamps_right_off.npy"]
"""Composite plate/new_2_lamps_top_off.npy",
    "Composite plate/new_3_lamps.npy",
    "Resin plates/circular_1_lamp_left_on.npy",
    "Resin plates/circular_1_lamp_right_on.npy",
    "Resin plates/circular_1_lamp_top_on.npy",
    "Resin plates/circular_2_lamps_left_off.npy",
    "Resin plates/circular_3_lamps.npy",
    "Resin plates/rec_1_lamp_left_on.npy",
    "Resin plates/rec_1_lamp_right_on.npy",
    "Resin plates/rec_2_lamps_left_off.npy",
    "Resin plates/rec_2 lamps_right_off.npy",
    "Resin plates/rec_2_lamps_top_off.npy",                
    "Resin plates/square_1_lamp_left_on.npy",
    "Resin plates/square_1_lamp_right_on.npy",
    "Resin plates/square_1_lamp_top_on.npy",                   
    "Resin plates/square_2_lamps_top_off.npy",
    "Resin plates/square_3_lamps.npy",
    "Resin plates/triangular_1_lamp_left_on.npy",              
    "Resin plates/triangular_1_lamp_top_on.npy",            
    "Resin plates/triangular_2_lamps_right_off.npy",
    "Resin plates/triangular_2_lamps_top_off.npy",
    "Resin plates/triangular_3_lamps.npy"]"""

TEST_DATA = [
    "Composite plate/new_1_lamp_left_on.npy",]
"""Composite plate/new_1_lamp_right_on.npy",
    "Composite plate/new_1_lamp_top_on.npy",
    "Resin plates/circular_2_lamps_right_off.npy",
    "Resin plates/circular_2_lamps_top_off.npy"
    "Resin plates/rec_1_lamp_top_on.npy",
    "Resin plates/rec_3_lamps.npy",
    "Resin plates/square_2_lamps_left_off.npy",
    "Resin plates/square_2_lamps_right_off.npy",
    "Resin plates/triangular_1_lamp_right_on.npy",
    "Resin plates/triangular_2_lamps_left_off.npy""" 

MASK_MAP = {
        "circular" : "Resin plates/circular.png",
        "rec" : "Resin plates/rec.png",
        "square" : "Resin plates/square.png",
        "tri" : "Resin plates/tri.png",
        "composite" : "Composite plate/composite plate mask.png"
}

NUM_PIXELS = 5000

class OneDCNN(torch.nn.Module):
    """
    not yet tested
    """
    def __init__(self, input_size, num_classes=2, num_hidden=128):
        super(OneDCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.pool = torch.nn.AdaptiveAvgPool1d(1)

        # fully connected dense layers
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(64, num_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )

        self.fc2 = torch.nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)  # flatten the output
        out = self.fc1(out)
        out = self.fc2(out)

        return out    

def train_model(model, train_loader, criterion, optimizer, num_epochs=10): 
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch, _ in train_loader:
            y_pred = model(X_batch)
            optimizer.zero_grad()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

def test_model(model, test_loader):
    model.eval()
    num_correct = 0
    for X_batch, y_batch, _ in test_loader:
        y_pred = model(X_batch)
        

def fuzzy_c_means(fft_data, mask):
    """
    not working 
    """
    # normalize data
    scaler = StandardScaler()
    fft_data_scaled = scaler.fit_transform(fft_data)
     
    print("mask shape: ", mask.shape)
    print("fft_data shape: ", fft_data_scaled.shape)

    # initialize centroids
    defect_pixels = fft_data_scaled[mask==0,:]
    non_defect_pixels = fft_data_scaled[mask==1,:]

    centroid_0 = np.mean(defect_pixels, axis=0)
    centroid_1 = np.mean(non_defect_pixels, axis=0)

    intial_centroids = np.vstack([centroid_0, centroid_1])
    
    v, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        fft_data_scaled.T, 
        c=2, m=2, 
        maxiter=1000, 
        error=0.005, 
        init= intial_centroids
    )
    
    # get the cluster labels by choosing the highest membership for each pixel
    fuzzy_labels = u.T
    clustered_data = np.argmax(fuzzy_labels, axis=1)
    
    return fuzzy_labels, clustered_data

def main(sim_data_path, exp_data_path):
    
    # initialize training dataset
    train_dataset = ThermalDataset(
        file_paths=TRAIN_DATA,
        data_dir=exp_data_path, 
        mask_map=MASK_MAP, 
        num_pixels=NUM_PIXELS, 
        add_zero_padding=True,
        apply_fft=True,
        apply_wiener=False,
        cutoff_frequency=0.1
    )
    
    # initialize testing dataset
    test_dataset = ThermalDataset(
        file_paths=TEST_DATA,
        data_dir=exp_data_path,
        mask_map=MASK_MAP,
        num_pixels=NUM_PIXELS, 
        add_zero_padding=True,
        apply_fft=True,
        apply_wiener=False,
        cutoff_frequency=0.1
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    X_train, y_train = [], []
    X_test, y_test = [], []

    model = OneDCNN(input_size=NUM_PIXELS, num_classes=2)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    # train data
    for batch_data in train_dataloader:
        filtered_fft, mask, freq = batch_data
        
        # format data
        freq = freq.squeeze(0)
        filtered_fft = np.abs(filtered_fft).squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()

        # randomly select pixels
        pixel_id = np.random.choice(NUM_PIXELS, 10)

        # plot randomly selected pixels
        p_fft = filtered_fft[:, pixel_id]
        plt.figure(figsize=(8, 4))
        plt.plot(freq[:700], np.abs(p_fft[:700]))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'FFT of Thermal Signal (Filtered) for Pixel {pixel_id}')
        plt.grid(True)
        plt.show()
        
        # turn into (num_pixels, num_frames)
        filtered_fft = filtered_fft.T

        X_train.append(filtered_fft)
        y_train.append(mask)
    
    # put train data into one array
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # train random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # check batch of data
    for batch_data in test_dataloader:
        filtered_fft, mask, freq = batch_data
        freq = freq.squeeze(0)
        filtered_fft = np.abs(filtered_fft).squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()

        # turn into (num_pixels, num_frames)
        filtered_fft = filtered_fft.T

        X_test.append(filtered_fft)
        y_test.append(mask)
    
    # put test data into one array
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    
    # predict on test data
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Random Forest Accuracy: {acc:.4f}") 
    # 0.3636 first run on 2+2 traintestvids
    # 0.4469 with 4+2 traintestvids

if __name__ == "__main__":
    # read file path for data
    sim_data_path = sys.argv[1]
    exp_data_path = sys.argv[2]
    main(sim_data_path, exp_data_path)
