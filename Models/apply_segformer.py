import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from thermal_dataset import ThermalDataset
from train_model import *
from segformer import *

# load model
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", ignore_mismatched_sizes=True, num_labels=2)
model.decode_head.classifier = nn.Conv2d(
    in_channels=256,
    out_channels=2,  # number of classes you want
    kernel_size=1
)
# define model path
weights_path = "/home/kpenners/teaminternship/segformer_sim_tenheat.pth"
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# define data path
data_dir = "/home/kpenners/teaminternship/Simulation Data/"

# preprocess frame for model input
def preprocess(frame, min, max):
    norm_frame = (frame - np.min(frame)) / (max - min)
    norm_frame = np.clip(norm_frame, 0, 1)
    norm_frame=norm_frame.astype(np.float32)
    norm_frame=np.squeeze(norm_frame)
    rgb = plt.cm.jet(norm_frame)[:, :,:3]
    norm_frame_uint8 = (rgb * 255).astype(np.uint8) 
    pil_img = Image.fromarray(norm_frame_uint8)

    return pil_img

# apply model to test simulation data
for file_path in TEST_SIM_DATA:
    print(f"\nprocessing: {file_path}")

    # load dataset for a video
    dataset = ThermalDataset(
        file_paths=[file_path],
        data_dir=data_dir,
        mask_dir ="masks/",
        sim=True
    )

    data, mask = dataset[0]

    data = data.reshape(-1, 480, 640)
    frames = data[::28]  # take every 28th frame

    preprocessed = [preprocess(frame, data.min(), data.max()) for frame in frames]

    with torch.no_grad():
        encoded = feature_extractor(preprocessed, return_tensors="pt")
        pixel_values = encoded["pixel_values"]

        # convert to PIL, resize label, repeat for batch, convert to tensor
        y_np = mask.astype(np.uint8)
        label_img = Image.fromarray(y_np)
        label_resized = label_img.resize((512, 512), resample=Image.NEAREST)
        label_tensor = torch.tensor(np.array(label_resized), dtype=torch.long).to(device)
        labels = label_tensor.unsqueeze(0).repeat(len(frames), 1, 1)  # shape: (B, H, W)
            
        # obtain predictions
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        logits_upsampled = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
        
        # softmax to get probabilities
        probs = torch.softmax(logits_upsampled, dim=1)

    # avg probs across frames
    avg_probs = probs.mean(dim=0)

    confidence_map = avg_probs[1]

    confidence_map_resized = F.interpolate(
        confidence_map.unsqueeze(0).unsqueeze(0), 
        size=(480, 640), 
        mode='bilinear', 
        align_corners=False
    ).squeeze().cpu().numpy()

    mask_resized = Image.fromarray(mask.astype(np.uint8)).resize((640, 480), resample=Image.NEAREST)
    arr = np.array(mask_resized)

    # plot confidence map and ground truth
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(arr, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    im = axes[1].imshow(confidence_map_resized, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Model Confidence (Class 1)")
    axes[1].axis("off")

    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # save confidence map
    file_name = file_path.split('/')[-1].replace('.npy', '.png')
    output_file = f"confidence_{file_name}"
    plt.savefig(output_file, dpi=300)
    print(f"saved confidence map to {output_file}")
    plt.close()