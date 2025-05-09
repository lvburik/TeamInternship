import os
import sys
import torch
from torch.utils.data import DataLoader
import joblib
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from thermal_dataset import ThermalDataset
import xgboost as xgb
from xgboost import XGBClassifier
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

class Network(torch.nn.Module): 
    def __init__(self, n_in, n_classes=1): 
        super(Network,self).__init__() 

        # convolutional and pooling blocks
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.upsample = torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=8, padding=1)
        self.final_conv = torch.nn.Conv2d(16, n_classes, kernel_size=1)

    
    def forward(self,x): 
        x = self.block1(x)
        x = self.block2(x)
        x = self.upsample(x)
        x = self.final_conv(x)

        # get output to exact original size
        x = F.interpolate(x, size=(480, 640), mode='bilinear', align_corners=False)

        return x  

def train_model(model, train_loader, criterion, optimizer, num_epochs=10): 
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            # format for model
            print("batch number: ", i)
            #print(f"X_batch shape: {X_batch.shape}")
            X_batch = X_batch.view(X_batch.shape[0], 103, 480, 640)
            #print(f"X_batch shape: {X_batch.shape}")
            y_batch = y_batch.view(y_batch.shape[0], 1, 480, 640).float()
            #print(f"y_batch shape: {y_batch.shape}")

            # obtain predictions
            y_pred = model(X_batch)
            #print(f"y_pred shape: {y_pred.shape}")

            # calculate loss, backprop
            optimizer.zero_grad()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print(f"[{epoch + 1}] Epoch Loss: {running_loss / len(train_loader):.4f}")

def test_model(model, test_loader):
    model.eval()
    num_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # format for model
            X_batch = X_batch.view(X_batch.shape[0], 103, 480, 640)
            #print(f"X_batch shape: {X_batch.shape}")
            y_batch = y_batch.view(y_batch.shape[0], 1, 480, 640).float()
            
            # obtain predictions
            y_pred = model(X_batch)
            predicted = (y_pred > 0.5).float() 
            
            # evaluate model
            y_true = y_batch.view(-1).cpu().numpy().astype(int) 
            y_pred_flat = predicted.view(-1).cpu().numpy().astype(int)
            evaluate(y_true, y_pred_flat)

def train_xgb_model(train_loader):
    X_train, y_train = [], []
    for fft_data, mask in train_loader:

        # format data
        fft_data = fft_data.squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()
        
        # turn into (num_pixels, num_freqs)
        fft_data = fft_data.T

        # turn into (num_pixels, )
        mask = mask.flatten()

        X_train.append(fft_data)
        y_train.append(mask)
        print(f"fft_data shape: {fft_data.shape}")

    # format data for rf
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="auc",
        learning_rate=0.1,
        n_estimators=100,
)
    model.fit(X_train, y_train)

    # save model
    joblib.dump(model, "xgb_model.joblib")
    print("xgb model saved")

    return model

def test_xgb_model(model, test_loader):
    X_test, y_test = [], []
    for fft_data, mask in test_loader:
        # format data
        fft_data = fft_data.squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()

        # turn into (num_pixels, num_freqs)
        fft_data = fft_data.T

        # turn into (num_pixels, )
        mask = mask.flatten()

        X_test.append(fft_data)
        y_test.append(mask)

    # format data for xgb
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # obtain predictions
    preds = model.predict(X_test)
    
    # evaluate xgb model
    print("xgb model evaluation")
    evaluate(y_test, preds)

    return preds

def train_rf_model(train_loader):
    # train random forst model
    X_train, y_train = [], []
    for fft_data, mask in train_loader:

        # format data
        fft_data = fft_data.squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()
        
        # turn into (num_pixels, num_freqs)
        fft_data = fft_data.T

        # turn into (num_pixels, )
        mask = mask.flatten()

        X_train.append(fft_data)
        y_train.append(mask)
        print(f"fft_data shape: {fft_data.shape}")

    # format data for rf
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    #x, y = resample(X_train, y_train, n_samples=100000, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # save model
    joblib.dump(rf, "rf_batch_rn.joblib")
    print("rf model saved")

    return rf

def test_rf_model(rf, test_loader):
    # test random forest model
    X_test, y_test = [], []
    for fft_data, mask in test_loader:
        # format data
        fft_data = fft_data.squeeze(0).numpy()
        mask = mask.squeeze(0).numpy()

        # turn into (num_pixels, num_freqs)
        fft_data = fft_data.T

        # turn into (num_pixels, )
        mask = mask.flatten()

        X_test.append(fft_data)
        y_test.append(mask)

    # format data for rf
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    # x, y = resample(X_test, y_test, n_samples=80000, random_state=42)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # obtain predictions
    preds = rf.predict(X_test)
    
    # evaluate rf model
    evaluate(y_test, preds)

    return preds

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

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")

    return acc, precision, recall, f1, iou


def main(sim_data_path, exp_data_path):
    
    # initialize training dataset
    train_dataset = ThermalDataset(
        file_paths=TRAIN_DATA,
        data_dir=exp_data_path, 
        mask_map=MASK_MAP,
        apply_PCA=False,
        extract_peaks=False,
        extract_patches=False,
        cutoff_frequency=1
    )
    
    # initialize testing dataset
    test_dataset = ThermalDataset(
        file_paths=TEST_DATA,
        data_dir=exp_data_path,
        mask_map=MASK_MAP,
        extract_peaks=False,
        extract_patches=False,
        cutoff_frequency=1
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    X_train, y_train = [], []
    X_test, y_test = [], []

    #freq = np.load(os.path.join(exp_data_path, "exp_freq.npy"))

    model = Network(n_in=103)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train model
    print("training model")

    # xgb
    model = train_xgb_model(train_dataloader)
    # torch.save(model, "xgb_model_rn.joblib")

    # cnn
    #train_model(model, train_dataloader, criterion, optimizer, num_epochs=5)
    #torch.save(model.state_dict(), "model5.pth")

    # rf
    #rf = train_rf_model(train_dataloader)
    
    # test model
    print("testing model")

    # xgb
    preds = test_xgb_model(model, test_dataloader)

    # rf
    #preds = test_rf_model(rf, test_dataloader)
    

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
