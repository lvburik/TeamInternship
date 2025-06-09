import sys
from torch.utils.data import DataLoader
import joblib
from Models.CNNandUNets.thermal_dataset import ThermalDataset
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from Models.CNNandUNets.train_model import *

train_model = 'rf' # choose 'rf' or 'xgb'

# train random forest model
def train_rf_model(train_loader):
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
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # if resampling instead of patching
    # X_train, y_train = resample(X_train, y_train, n_samples=300000, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    # save model
    joblib.dump(rf, "rf_patch.joblib")
    print("rf model saved")

    return rf

# test random forest model
def test_rf_model(rf, test_loader):
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

    # if resampling instead of patching
    # X_test, y_test = resample(X_test, y_test, n_samples=100000, random_state=42)

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # obtain predictions
    preds = rf.predict(X_test)
    
    # evaluate rf model
    evaluate(y_test, preds)

    return preds

# train xgb model (not included in final report)
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

    # format data for rf
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int32)

    X_train, y_train = resample(X_train, y_train, n_samples=10000, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.01,
        n_estimators=100,
        scale_pos_weight=5,
        early_stopping_rounds=10,
        tree_method='hist'
    )
    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_val, y_val)],
              verbose=True)
    print('fitted xgb model')

    # save model
    joblib.dump(model, "xgb_patches.joblib")
    print("xgb model saved")

    return model

# test xgb model (not included in final report)
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


def main(sim_data_path, exp_data_path):
    
    # initialize training dataset
    train_dataset = ThermalDataset(
        file_paths=TRAIN_DATA,
        data_dir=exp_data_path, 
        mask_map=MASK_MAP,
        mask_dir=mask_dir,
        extract_patches=True,   # change if resampling pixels
        sim=False               # change if using sim data
    )
    
    # initialize testing dataset
    test_dataset = ThermalDataset(
        file_paths=TEST_DATA,
        data_dir=exp_data_path,
        mask_map=MASK_MAP,
        mask_dir=mask_dir,
        extract_patches=True,   # change if resampling pixels
        sim=False               # change if using sim data
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    if train_model == 'xgb':
        # train xgb model
        print("training xgb...")
        model = train_xgb_model(train_dataloader)

        # test xgb model
        print("testing xgb...")
        preds = test_xgb_model(model, test_dataloader)
    elif train_model == 'rf':
        # train rf model
        print("training random forest...")
        rf = train_rf_model(train_dataloader)

        # test rf model
        print("testing random forest...")
        preds = test_rf_model(rf, test_dataloader)

if __name__ == "__main__":
    # read file path for data
    #   first argument should lead to simulation data directory
    #   second should lead to experimental data directory
    sim_data_path = sys.argv[1]
    exp_data_path = sys.argv[2]
    main(sim_data_path, exp_data_path)
