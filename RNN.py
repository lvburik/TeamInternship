# %%
from Models import thermal_dataset
from Preprocessing import *
import os


# %% select data directory

data_dir = "C:/Users/20202555/OneDrive - TU Eindhoven/Documents/M AI&ES - Year 1/Team-internship/Experimental_Data/Resin/"

filepaths = iles = [
    f
    for f in os.listdir(data_dir)[1:]
    if os.path.isfile(os.path.join(data_dir, f))
]

# %% define a cropper for the video and label
import random

class RandomCropVideoAndLabel:
    def __init__(self, crop_size = (20, 20)):
        self.crop_h, self.crop_w = crop_size

    def __call__(self, video, label):
        video = video[:1300]  # select first 1300 frames
        # reshape video to (time, height, width)
        video = video.reshape(video.shape[0], 480, 640)

        t, h, w = video.shape
        assert label.shape == (h, w)

        if h < self.crop_h or w < self.crop_w:
            raise ValueError("Crop size must be smaller than video dimensions.")

        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)

        

        video_crop = video[:, top:top+self.crop_h, left:left+self.crop_w].reshape((t, self.crop_h * self.crop_w))
        label_crop = label[top:top+self.crop_h, left:left+self.crop_w]

        return video_crop, label_crop
    
# %%
Dataset = thermal_dataset.ThermalDataset(filepaths, 
                          data_dir=data_dir,
                          mask_map={
                              
                            "circular" : "circular.png",
                              
                          },
                          center_data=True,
                          add_zero_padding=False,
                          apply_fft=False,
                          cutoff_frequency=1,
                          apply_PCA=False,
                          extract_patches=False,
                          extract_cnn_patches=False,
                          transform=RandomCropVideoAndLabel()
                          )

# %%
Video, Label = Dataset[0]

# %% define the LTSTM model
import torch
from torch import nn
from torch.utils.data import DataLoader

class VideoLSTM(nn.Module):
    def __init__(self, input_dim=400, hidden_dim=256, num_layers=2, output_dim=400):
        super(VideoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, 1300, 400]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, 1300, hidden_dim]

        # Take only the last time step output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        output = self.fc(last_output)     # [batch_size, 400]
        output = output.view(-1, 20, 20)  # Reshape to [batch_size, 20, 20]
        return output
    
#set pu dataloader
Loader = DataLoader(Dataset, batch_size=1, shuffle=True)

# %% initialize the model and optimizer
model = VideoLSTM(input_dim=400, hidden_dim=256, num_layers=2, output_dim=400)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# %% define the loss function
criterion = nn.BCEWithLogitsLoss()

# %% training loop
num_epochs = 1
for epoch in range(num_epochs):
    for i, (video, label) in enumerate(Loader):

        
        video = video.float()  # ensure video is float
        label = label.float()  # ensure label is float

        optimizer.zero_grad()  # zero the gradients

        output = model(video)  # forward pass
        loss = criterion(output, label)  # compute loss

        loss.backward()  # backward pass
        optimizer.step()  # update weights

        if (i + 1) % 10 == 0:  # print every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(Loader)}], Loss: {loss.item():.4f}")





# %%
