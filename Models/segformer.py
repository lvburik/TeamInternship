import numpy as np
import glob
import os
from PIL import Image
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import piq
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F

import sys
from thermal_dataset import ThermalDataset
from train_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATA = [
    "Composite plate/new_2_lamps_left_off.npy",
    "Composite plate/new_2_lamps_right_off.npy",
    "Composite plate/new_2_lamps_top_off.npy",
    "Composite plate/new_3_lamps.npy",
    "Resin plates/circular_1_lamp_left_on.npy",
    "Resin plates/circular_1_lamp_right_on.npy",
    "Resin plates/circular_1_lamp_top_on.npy",
    "Resin plates/circular_2_lamps_left_off.npy",
    "Resin plates/circular_3_lamps.npy",
    "Resin plates/rec_1_lamp_left_on.npy",
    "Resin plates/rec_1_lamp_right_on.npy",
    "Resin plates/rec_2_lamps_left_off.npy",
    "Resin plates/rec_2_lamps_right_off.npy",
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
    "Resin plates/triangular_3_lamps.npy"]

def preprocess(frame, min, max):
    norm_frame = (frame - min) / (max - min)
    norm_frame = norm_frame.clamp(0, 1)
    norm_frame = (norm_frame * 255).byte().cpu().numpy()
    img = Image.fromarray(norm_frame)
    return img.convert("RGB")


def main(sim_data_path, exp_data_path):
    
    train_dataset = ThermalDataset(
        file_paths=TRAIN_DATA,
        data_dir=exp_data_path, 
        mask_map=MASK_MAP,
    )

    test_dataset = ThermalDataset(
        file_paths=TEST_DATA,
        data_dir=exp_data_path, 
        mask_map=MASK_MAP,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # sample frames

    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
    model.decode_head.classifier = nn.Conv2d(
        in_channels=256,
        out_channels=2,  # number of classes you want
        kernel_size=1
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(15):
        total_loss = 0
        counter=1
        for (X_batch, y_batch) in train_dataloader:
            print(f"Batch {counter} of Epoch {epoch+1}")
            print(f"X_batch is {X_batch.shape}") # (frames, 307200)
            print(f"y_batch is {y_batch.shape}")

            X_batch = X_batch.to(device).squeeze(0)  # (1, frames, 480, 640)
            X_batch = X_batch.reshape(-1, 480, 640)
            y_batch = y_batch.to(device).squeeze(0)

            print(f"X_batch after reshape is {X_batch.shape}")

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

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            logits_upsampled = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

            loss = criterion(logits_upsampled, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")
            total_loss += loss.item()
            counter += 1

        print(f"Epoch {epoch+1}, Average Loss: {total_loss/counter:.4f}")

    torch.save(model.state_dict(), "segformer_15ep.pth")


if __name__ == "__main__":
    # read file path for data
    sim_data_path = sys.argv[1]
    exp_data_path = sys.argv[2]
    main(sim_data_path, exp_data_path)




