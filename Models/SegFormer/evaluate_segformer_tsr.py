import numpy as np
import os
import glob
from PIL import Image
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, Trainer, TrainingArguments
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
import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.polynomial.polynomial import Polynomial

# data
from Resin_plates import *
from data import *


#Import model

feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", ignore_mismatched_sizes=True, num_labels=2)
model.decode_head.classifier = nn.Conv2d(
    in_channels=256,
    out_channels=2,  
    kernel_size=1
)
weights_path = "segformer_tsr_25ep.pth"
state_dict = torch.load(weights_path, map_location="cpu")  

model.load_state_dict(state_dict)
model.eval()

video=np.load("circular_2_lamps_right_off.npy")
video_files=[video]

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

sampling_rate=50

#Preprocess

def TSR(video):
    filtered_data = gaussian_filter(video, sigma=2)

    log_normalized_data = np.log(np.clip(filtered_data, 1e-8, None))

    d = log_normalized_data.shape[0]
    t = np.linspace(0.5, d * 0.5, d)
    log_t = np.log(t)

    n_pixels = video.shape[1]
    first_derivatives = np.empty((n_pixels, d), dtype=np.float32)

    for i in range(n_pixels):
        y = log_normalized_data[:, i]
        coeffs = np.polyfit(t, y, 4)
        poly_vals = np.polyval(coeffs, t)
        first_derivative = np.gradient(poly_vals, t)
        first_derivatives[i] = first_derivative

    return first_derivatives


video_channels=[]
frames=[]
first_frames=[]
second_frames=[]
for video in video_files:
    first=TSR(video)
    print(f"first dev is {first}")
    original=video.reshape(-1, 480, 640)
    first=first.T
    first = first.reshape(-1, 480, 640)
    sampled_first_frames = first[::sampling_rate]
    sampled_first_frames=sampled_first_frames[:20]
    
INPUT_SIZE = 512 
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

tri=label_tensor(mask_t)

def dataset(frames, video):
    list_frames=[]
    for f in frames:
        pr_frame=preprocess(f, video.min(), video.max())
        list_frames.append(pr_frame)
    encoded_frames=feature_extractor(list_frames, return_tensors="pt")
    return encoded_frames

validation_dataset=dataset(sampled_first_frames, video)
print(validation_dataset)


#Evaluate

with torch.no_grad():
    outputs = model(validation_dataset["pixel_values"])
    
logits = outputs.logits  
logits_upsampled = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
predicted_mask = logits_upsampled.argmax(dim=1) 

predicted_mask=predicted_mask.float()
predicted_mask_4d = F.interpolate(predicted_mask[15].unsqueeze(0).unsqueeze(0), size=(480, 640), mode='bilinear', align_corners=False)
predicted_mask_resized=predicted_mask_4d.squeeze(0).squeeze(0)

#Visualize
plt.imshow(predicted_mask_resized.cpu(), cmap='gray')  
plt.title('Predicted mask')
plt.axis('off')  
plt.show()