import sys
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn.functional as F

from thermal_dataset import ThermalDataset
from train_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define training data paths within data dir
TRAIN_DATA = [
    "raw data/Composite plate/new_1_lamp_left_on.npy",
    "raw data/Composite plate/new_1_lamp_right_on.npy",
    "raw data/Composite plate/new_1_lamp_top_on.npy",
    "raw data/Composite plate/new_2_lamps_left_off.npy",
    "raw data/Resin plates/circular_1_lamp_left_on.npy",
    "raw data/Resin plates/circular_1_lamp_right_on.npy",
    "raw data/Resin plates/circular_1_lamp_top_on.npy",
    "raw data/Resin plates/circular_2_lamps_left_off.npy",
    "raw data/Resin plates/circular_3_lamps.npy",
    "raw data/Resin plates/rec_1_lamp_left_on.npy",
    "raw data/Resin plates/rec_1_lamp_right_on.npy",
    "raw data/Resin plates/rec_2_lamps_left_off.npy",
    "raw data/Resin plates/rec_2_lamps_right_off.npy",
    "raw data/Resin plates/rec_2_lamps_top_off.npy",                
    "raw data/Resin plates/square_1_lamp_left_on.npy",
    "raw data/Resin plates/square_1_lamp_right_on.npy",
    "raw data/Resin plates/square_1_lamp_top_on.npy",                   
    "raw data/Resin plates/square_2_lamps_top_off.npy",
    "raw data/Resin plates/square_3_lamps.npy",
    "raw data/Resin plates/triangular_1_lamp_left_on.npy",              
    "raw data/Resin plates/triangular_1_lamp_top_on.npy",            
    "raw data/Resin plates/triangular_2_lamps_right_off.npy",
    "raw data/Resin plates/triangular_2_lamps_top_off.npy",
    "raw data/Resin plates/triangular_3_lamps.npy"]

# define test data paths within data dir
TEST_DATA = [
    "raw data/Composite plate/new_2_lamps_right_off.npy",
    "raw data/Composite plate/new_2_lamps_top_off.npy",
    "raw data/Composite plate/new_3_lamps.npy",
    "raw data/Resin plates/circular_2_lamps_right_off.npy",
    "raw data/Resin plates/circular_2_lamps_top_off.npy",
    "raw data/Resin plates/rec_1_lamp_top_on.npy",
    "raw data/Resin plates/rec_3_lamps.npy",
    "raw data/Resin plates/square_2_lamps_left_off.npy",
    "raw data/Resin plates/square_2_lamps_right_off.npy",
    "raw data/Resin plates/triangular_1_lamp_right_on.npy",
    "raw data/Resin plates/triangular_2_lamps_left_off.npy"]

# preprocess frames for model input
def preprocess(frame, min, max):
    norm_frame = (frame - min) / (max - min)
    norm_frame = norm_frame.clamp(0, 1)
    norm_frame = (norm_frame * 255).byte().cpu().numpy()
    img = Image.fromarray(norm_frame)
    return img.convert("RGB")

def train_segformer(model, feature_extractor, dataloader, num_epochs=15):
    modelname = "segformer_every10thframe_50ep"
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for (X_batch, y_batch) in dataloader:

            X_batch = X_batch.to(device).squeeze(0)  # (1, frames, 480, 640)
            X_batch = X_batch.reshape(-1, 480, 640)
            y_batch = y_batch.to(device).squeeze(0)

            frames = X_batch[::20]

            # preprocess and convert to list of PIL images
            preprocessed = [preprocess(frame, X_batch.min(), X_batch.max()) for frame in frames]

            # convert to model input
            encoded = feature_extractor(preprocessed, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)
            
            # convert to PIL, resize label, repeat for batch, convert to tensor
            y_np = y_batch.cpu().numpy().astype(np.uint8)
            label_img = Image.fromarray(y_np)
            label_resized = label_img.resize((512, 512), resample=Image.NEAREST)
            label_tensor = torch.tensor(np.array(label_resized), dtype=torch.long).to(device)
            labels = label_tensor.unsqueeze(0).repeat(len(frames), 1, 1)  # shape: (B, H, W)

            # obtain predictions
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            logits_upsampled = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

            loss = criterion(logits_upsampled, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    # save model
    torch.save(model.state_dict(), f"{modelname}.pth")
    print(f"{modelname} saved")

    # save loss over epochs
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss")

    plt.savefig(f"training_loss_{modelname}.png")
    plt.close()

    return model

def test_segformer(model, feature_extractor, dataloader):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # format for model
            X_batch = X_batch.to(device).squeeze(0)
            X_batch = X_batch.reshape(-1, 480, 640)
            y_batch = y_batch.to(device).squeeze(0)
            
            frames = X_batch[::28]

            # preprocess and convert to list of PIL images
            preprocessed = [preprocess(frame, X_batch.min(), X_batch.max()) for frame in frames]

            # convert to model input
            encoded = feature_extractor(preprocessed, return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)

            # convert to PIL, resize label, repeat for batch, convert to tensor
            y_np = y_batch.cpu().numpy().astype(np.uint8)
            label_img = Image.fromarray(y_np)
            label_resized = label_img.resize((512, 512), resample=Image.NEAREST)
            label_tensor = torch.tensor(np.array(label_resized), dtype=torch.long).to(device)
            labels = label_tensor.unsqueeze(0).repeat(len(frames), 1, 1)  # shape: (B, H, W)
            
            # obtain predictions
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            logits_upsampled = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
            preds = logits_upsampled.argmax(dim=1)

            # flatten and store for evaluation
            y_true.append(labels.view(-1).cpu().numpy())
            y_pred.append(preds.view(-1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    evaluate(y_true, y_pred)

def main(sim_data_path, exp_data_path):

    train_dataset = ThermalDataset(
        file_paths=TRAIN_DATA,
        data_dir=exp_data_path, 
        mask_map = MASK_MAP,
    )

    test_dataset = ThermalDataset(
        file_paths=TEST_DATA,
        data_dir=exp_data_path, 
        mask_map = MASK_MAP,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # initialize model
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
    model.decode_head.classifier = nn.Conv2d(
        in_channels=256,
        out_channels=2,  # number of classes you want
        kernel_size=1
    )
    model.to(device)

    # train model
    print("training segformer...")
    model = train_segformer(model, feature_extractor, train_dataloader, num_epochs=50)
    
    # for loading existing model
    """weights_path = "segformer_15ep.pth"
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)"""

    # testing model
    print("testing segformer...")
    test_segformer(model, feature_extractor, test_dataloader)

if __name__ == "__main__":
    # read file path for data
    sim_data_path = sys.argv[1]
    exp_data_path = sys.argv[2]
    main(sim_data_path, exp_data_path)




