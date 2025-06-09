import numpy as np
import sys
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, logging
import warnings

# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ignore transformer warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# path to saved trained model
model_path = "SegFormer/segformer_50ep.pth"

# load segformer model
def load_model(model_path):
    # load feature extractor and model required for prediction
    feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", 
                                                             ignore_mismatched_sizes=True, num_labels=2)
    model.decode_head.classifier = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)

    # define model path
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, feature_extractor

# preprocess frame for model input
def preprocess(frame, min, max):
    norm_frame = (frame - min) / (max - min)
    norm_frame = np.clip(norm_frame, 0, 1)
    norm_frame=norm_frame.astype(np.float32)
    norm_frame=np.squeeze(norm_frame)
    rgb = plt.cm.jet(norm_frame)[:, :,:3]
    norm_frame_uint8 = (rgb * 255).astype(np.uint8) 
    pil_img = Image.fromarray(norm_frame_uint8)

    return pil_img

def run_inference(video_path):
    # load model
    print("Loading segformer...")
    segformer, processor = load_model(model_path)

    # load data
    print("Loading thermal video...")
    data = np.load(video_path, mmap_mode='r')

    # reshape data
    data = data.reshape(-1, 480, 640)

    # sample frames from thermal video for prediction
    frames = data[::10]

    # preprocess frames
    min = data.min()
    max = data.max()
    preprocessed = [preprocess(frame, min, max) for frame in frames]

    with torch.no_grad():
        encoded = processor(preprocessed, return_tensors="pt")
        pixel_values = encoded["pixel_values"]
            
        # obtain predictions
        outputs = segformer(pixel_values=pixel_values)
        logits = outputs.logits
        logits_upsampled = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
        
        # softmax to get probabilities
        probs = torch.softmax(logits_upsampled, dim=1)
    
    # take average across all predicted frames
    avg_probs = probs.mean(dim=0)
    confidence_map = avg_probs[1]

    # resize into original input size
    confidence_map_resized = F.interpolate(
        confidence_map.unsqueeze(0).unsqueeze(0), 
        size=(480, 640), 
        mode='bilinear', 
        align_corners=False
    ).squeeze().cpu().numpy()

    # visualize predictions
    plt.figure(figsize=(8, 6))
    plt.imshow(confidence_map_resized, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label="Model Confidence")
    plt.axis("off")
    plt.tight_layout()
   
    # save prediction plot
    file_name = os.path.basename(video_path).replace('.npy', '.png')
    output_file = f"pred_{file_name}"
    plt.savefig(output_file, dpi=300)
    print(f"Saved prediction to {output_file}")

    plt.close()     

if __name__ == "__main__":
    video_path = sys.argv[1]
    run_inference(video_path)
