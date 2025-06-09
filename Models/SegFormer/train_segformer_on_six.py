# %% imports
import numpy as np
import os
import glob
from PIL import Image
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import piq

#torch imports
import torch
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F
from torchmetrics import JaccardIndex
from segmentation_models_pytorch.losses import DiceLoss

# data
from Resin_plates import *
from data import *


#############################Import data

resin_folder="data"
r_data=glob.glob(os.path.join(resin_folder,"*.npy"))
resin_data=[]
print("data is loaded")

path = "Resin_plates/circular.png"
image=Image.open(path)
mask_c=np.array(image)
#print(mask_c.shape)

path_c = "Resin_plates/rec.png"
image_c=Image.open(path_c)
mask_rec=np.array(image_c)
#print(mask_rec.shape)

path_s = "Resin_plates/square.png"
image_s=Image.open(path_s)
mask_s=np.array(image_s)
#print(mask_s.shape)

path_t = "Resin_plates/tri.png"
image_t=Image.open(path_t)
mask_t=np.array(image_t)
#print(mask_t.shape)

print("Masks are loaded")

video_one = np.load(r_data[0]).reshape(-1, 480, 640)
video_two = np.load(r_data[1]).reshape(-1, 480, 640)
video_three = np.load(r_data[2]).reshape(-1,480, 640)
video_four = np.load(r_data[3]).reshape(-1,480, 640)
video_five = np.load(r_data[4]).reshape(-1,480, 640)
video_six = np.load(r_data[5]).reshape(-1,480, 640)

video_files = [video_one, video_two, video_three, video_four, video_five,video_six]  
masks=[mask_c, mask_rec, mask_s, mask_rec]
frames=[]

print("videos are loaded")

for video in video_files:
    sampling_rate = 28
    sampled_frames = video[::sampling_rate]
    frames.append(sampled_frames)
print(f" First dimension is {len(frames)}")
print(f" Second dimension is {len(frames[0])}")

all_frames=[item for sublist in frames for item in sublist]
print(f"Total number of frames is {len(all_frames)}")

##############################################################Import model
INPUT_SIZE = 512 
temp_min = 20.82
temp_max = 31.68
input_size = 256
batch_size = 16
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True)
model.decode_head.classifier = nn.Conv2d(
    in_channels=256,
    out_channels=2, 
    kernel_size=1
)
model.to(device)


###########################################################Preprocess data

def preprocess(frame, min, max):
    norm_frame = (frame - np.min(frame)) / (max - min)
    norm_frame = np.clip(norm_frame, 0, 1)
    norm_frame=norm_frame.astype(np.float32)
    norm_frame=np.squeeze(norm_frame)
    rgb = plt.cm.jet(norm_frame)[:, :,:3]
    norm_frame_uint8 = (rgb * 255).astype(np.uint8) 
    pil_img = Image.fromarray(norm_frame_uint8)
    return pil_img


def label_tensor(mask):
    label = Image.fromarray(mask.astype(np.uint8))
    mask_resized = label.resize((512, 512), resample=Image.NEAREST) 
    label_tensor = torch.tensor(np.array(mask_resized), dtype=torch.long)
    label_tensor=label_tensor.unsqueeze(0)
    return label_tensor

circle=label_tensor(mask_c)
rec=label_tensor(mask_rec)
square=label_tensor(mask_s)
tri=label_tensor(mask_t)

def dataset(frames, video):
    list_frames=[]
    for f in frames:
        pr_frame=preprocess(f, video.min(), video.max())
        list_frames.append(pr_frame)
    encoded_frames=feature_extractor(list_frames, return_tensors="pt")
    v1 = circle.repeat(len(frames[0]), 1, 1) 
    v2 = circle.repeat(len(frames[0]), 1, 1)
    v3 =  rec.repeat(len(frames[0]), 1, 1) 
    v4 = square.repeat(len(frames[0]), 1, 1)
    v5 = square.repeat(len(frames[0]), 1, 1)
    v6 = tri.repeat(len(frames[0]), 1, 1) 
    all_labels = torch.cat([v1, v2, v3, v4, v5, v6], dim=0)
    encoded_frames["labels"]=all_labels
    return encoded_frames


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

#####################################################################Train data

model.train()

unloaded_dataset=dataset(all_frames, video_one)
train_dataset = ImageDataset(unloaded_dataset['pixel_values'], unloaded_dataset['labels'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    total_loss = 0
    counter=1
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        #print(f"pixel values are {pixel_values.shape}")
        labels = batch["labels"].to(device)
        #print(f"labels are {labels.shape}")
        outputs = model(pixel_values=pixel_values, labels=labels)
        #print(f"outputs are {outputs.shape}")
        logits=outputs.logits
        logits_upsampled = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
        metric=nn.CrossEntropyLoss(reduction="mean")
        loss=metric(logits_upsampled,labels)
        print(f"Batch {counter} of Epoch {epoch+1} - loss is {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter+=1
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/18:.4f}")
torch.save(model.state_dict(), "segformer_weights_for_15_epochs_all.pth")










