import os
import sys
import torch
from torch.utils.data import DataLoader
import torchmetrics
from sklearn.preprocessing import StandardScaler
from thermal_dataset import ThermalDataset
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

MASK_MAP = {
        "circular" : "Resin plates/circular.png",
        "rec" : "Resin plates/rec.png",
        "square" : "Resin plates/square.png",
        "tri" : "Resin plates/tri.png",
        "composite" : "Composite plate/composite plate mask.png"
}

NUM_PIXELS = 307200

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

def calculate_iou(y_true, y_pred):
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    iou = tp / (tp + fn + fp) if tp + fn + fp > 0 else 0
    return iou

def evaluate_rf(y_true, y_pred):

    # caculate evaluation metrics
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
        extract_peaks=False,
        cutoff_frequency=0.1
    )
    
    # initialize testing dataset
    test_dataset = ThermalDataset(
        file_paths=TEST_DATA,
        data_dir=exp_data_path,
        mask_map=MASK_MAP,
        num_pixels=NUM_PIXELS, 
        extract_peaks=False,
        cutoff_frequency=0.1
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    X_train, y_train = [], []
    X_test, y_test = [], []

    model = OneDCNN(input_size=NUM_PIXELS, num_classes=2)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #freq = np.load(os.path.join(exp_data_path, "exp_freq.npy"))
    
    # train data
    for batch_data in train_dataloader:
        filtered_fft, mask = batch_data

        # format data
        filtered_fft = np.abs(filtered_fft).squeeze(0).numpy()
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
        filtered_fft = filtered_fft.T

        # turn into (num_pixels, )
        mask = mask.flatten()

        X_train.append(filtered_fft)
        y_train.append(mask)
    
    # put train data into one array
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    x, y = resample(X_train, y_train, n_samples=300000, random_state=42)

    print("training random forest now")
    # train random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x, y)

    # check batch of data
    for batch_data in test_dataloader:
        filtered_fft, mask = batch_data
        
        # req = freq.squeeze(0)
        filtered_fft = np.abs(filtered_fft).squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()

        # turn into (num_pixels, num_frames)
        filtered_fft = filtered_fft.T

        # turn into (num_pixels, )
        mask = mask.flatten()

        X_test.append(filtered_fft)
        y_test.append(mask)
    
    # put test data into one array
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    x, y = resample(X_test, y_test, n_samples=80000, random_state=42)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # predict on test data
    preds = rf.predict(x)

    # calculate evaluation metrics
    acc, precision, recall, f1, iou = evaluate_rf(y, preds)
    print("Random Forest Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")
   
    # random forest accuracy run on fft data with 100k samples
    #   0.904

    # Random Forest Evaluation: trained with 10k samples
    #   Accuracy: 0.9105
    #   Precision: 0.9258
    #   Recall: 0.9809
    #   F1 Score: 0.9525
    #   IoU: 0.9094

    # Random Forest Evaluation: trained with 300k samples
    #   Accuracy: 0.9056
    #   Precision: 0.9323
    #   Recall: 0.9667
    #   F1 Score: 0.9492
    #   IoU: 0.9033

if __name__ == "__main__":
    # read file path for data
    sim_data_path = sys.argv[1]
    exp_data_path = sys.argv[2]
    main(sim_data_path, exp_data_path)
