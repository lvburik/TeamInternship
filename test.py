from CompositePlate import *
from Masks import *
from ResinPlates import *
import numpy as np
import glob
import os
from PIL import Image
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import matplotlib.pyplot as plt
#import transformers
from transformers import Trainer, TrainingArguments
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import piq
from torchmetrics.functional import structural_similarity_index_measure as ssim






resin_folder="CompositePlate"
r_data=glob.glob(os.path.join(resin_folder,"*.npy"))
resin_data=[]
print("a is done")


path = "Masks/Resin plates/circular.png"
image=Image.open(path)
mask=np.array(image)
#print(mask.shape)

# Assume video shape (1443, 480, 640)
video_one = np.load(r_data[0]).reshape(-1, 480, 640)
second=np.load(r_data[1])
#print(second.shape)
video_two= np.load(r_data[1]).reshape(-1, 480, 640)



# Calculate min/max temps for normalization (per video)
TEMP_MIN = video_one.min()
TEMP_MAX = video_one.max()

sampling_rate = 5
sampled_frames = video_one[::sampling_rate]
sampled_frames=sampled_frames[:288]
#print(len(sampled_frames))
#print(sampled_frames[0])

INPUT_SIZE = 512 



############################################################################
video_files = [video_one, video_two]  
mask_files = [mask, mask]   

temp_min = 20.82
temp_max = 31.68
sampling_rate = 5
input_size = 256
batch_size = 16
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#print(sampled_frames[0])


def preprocess(frame):
    norm_frame = (frame - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
#print(f"After normalization: {norm_frame}")
    norm_frame = np.clip(norm_frame, 0, 1)
#print(f"After clipping: {norm_frame}")
    norm_frame=norm_frame.astype(np.float32)
#print(f"New one: {norm_frame}")
    norm_frame=np.squeeze(norm_frame)
    rgb = plt.cm.jet(norm_frame)[:, :,:3]
    norm_frame_uint8 = (rgb * 255).astype(np.uint8) 
    pil_img = Image.fromarray(norm_frame_uint8)
#print(rgb)
#print(len(rgb))
#print(len(rgb[0]))
    return pil_img

print(preprocess(sampled_frames[0]))

model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
batch_size = 16
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
model.decode_head.classifier = nn.Conv2d(
    in_channels=256,
    out_channels=2,  # number of classes you want
    kernel_size=1
)
model.to(device)
#print(model)

label = Image.fromarray(mask.astype(np.uint8))
mask_resized = label.resize((512, 512), resample=Image.NEAREST)
label_tensor = torch.tensor(np.array(mask_resized), dtype=torch.long)
label_tensor=label_tensor.unsqueeze(0)

def dataset(frames):
    list_frames=[]
    for f in frames:
        pr_frame=preprocess(f)
        list_frames.append(pr_frame)
    encoded_frames=feature_extractor(list_frames, return_tensors="pt")
    encoded_frames["labels"] = label_tensor.repeat(len(frames), 1, 1) 
        
    return encoded_frames

result=dataset(sampled_frames)
print(result["pixel_values"].shape)
print(result["labels"].shape)
print(result)


class ImageDataset(Dataset):
    def __init__(self, pixel_values, labels):
        self.pixel_values = pixel_values
        self.labels = labels

    def __len__(self):
        return self.pixel_values.shape[0]

    def __getitem__(self, idx):
        return {
            'pixel_values': self.pixel_values[idx],
            'labels': self.labels[idx]
        }



optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
unloaded_dataset=dataset(sampled_frames)
train_dataset = ImageDataset(unloaded_dataset['pixel_values'], unloaded_dataset['labels'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device) # (B, 3, H, W)
        print(f"pixel values are {pixel_values}")
        labels = batch["labels"].to(device)# (B, H, W)
        print(f"labels are {labels}")
        outputs = model(pixel_values=pixel_values, labels=labels)
        print(f"outputs are {outputs}")
        logits=outputs.logits
        print(f"logits are {logits}")
        loss = 1- ssim(logits,labels)
        print(f"loss is {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(sampled_frames):.4f}")