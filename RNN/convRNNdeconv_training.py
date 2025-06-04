# %%
import RNN_data
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import recall, precision, f1_score
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn

# %%
dataset_Path = 'compacted_improved.h5'


# %% create a dataloader
dataloader = RNN_data.create_dataloader(dataset_Path, video_types = ('circular', 'square', 'rec'), resolution=(480, 640), crop_size=(20, 20), batch_size=32)


# %% define model and submodels
class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(CNNEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 20x20 → 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 20x20
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 20x20
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # [B*T, 128, 1, 1]
        self.flatten = nn.Flatten()               # [B*T, 128]

    def forward(self, x):  # x: [B*T, 1, 20, 20]
        x = self.cnn(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x  # [B*T, 128]


class RNNTemporalModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(RNNTemporalModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):  # x: [B, T, 128]
        _, (h_n, _) = self.rnn(x)
        return h_n[-1]  # Final hidden state: [B, 256]


class Decoder(nn.Module):
    def __init__(self, input_dim=256):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(input_dim, 64 * 5 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 10x10
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 20x20
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),  # 20x20
            nn.Sigmoid()  # Ensure output is in [0,1]
        )

    def forward(self, x):  # x: [B, 256]
        x = self.fc(x)              # [B, 64*5*5]
        x = x.view(-1, 64, 5, 5)    # [B, 64, 5, 5]
        return self.deconv(x)       # [B, 1, 20, 20]


class VideoToImageGrayscale(nn.Module):
    def __init__(self):
        super(VideoToImageGrayscale, self).__init__()
        self.encoder = CNNEncoder()
        self.temporal = RNNTemporalModel()
        self.decoder = Decoder()

    def forward(self, video):  # video: [B, T, H, W] with no channel dim
        B, T, H, W = video.size()

        # Add channel dim: [B, T, H, W] → [B, T, 1, H, W]
        video = video.unsqueeze(2)

        # Flatten for CNN encoder: [B*T, 1, H, W]
        video = video.view(B * T, 1, H, W)
        feats = self.encoder(video)        # [B*T, 128]

        # Reshape for RNN: [B, T, 128]
        feats = feats.view(B, T, -1)
        temporal = self.temporal(feats)    # [B, 256]

        # Decode into grayscale image: [B, 1, 20, 20] → [B, 20, 20]
        out = self.decoder(temporal)      # [B, 1, 20, 20]
        return out.squeeze(1)             # [B, 20, 20]

# %% define the model
Model = VideoToImageGrayscale()

# %% check the model
print(Model)


# %% training setup
import torch.optim as optim

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = Model.to(device)

# SSIM loss
def ssim_loss(pred, target):
    # pred, target: [B, 1, H, W], values in [0,1]
    loss = 0
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    for i in range(pred.shape[0]):
        loss += 1 - ssim(pred[i,0], target[i,0], data_range=1)
    return loss / pred.shape[0]

# Optimizer
optimizer = optim.Adam(Model.parameters(), lr=1e-3)

# %% Training loop
epochs = 1
Model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        video, label = batch  # video: [B, T, 1, 20, 20], label: [B, 1, 20, 20]
        video = video.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)
        optimizer.zero_grad()
        output = Model(video)
        loss = ssim_loss(output, label)
        torch.tensor(loss, requires_grad=True).backward()
        optimizer.step()
        total_loss += loss
        # Print progress every 10% of batches
        if (batch_idx + 1) % max(1, len(dataloader) // 10) == 0 or (batch_idx + 1) == len(dataloader):
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, SSIM Loss: {loss:.4f}")
    # Print at the end of each epoch
    print(f"Epoch {epoch+1}/{epochs} completed, Avg SSIM Loss: {total_loss/len(dataloader):.4f}")

# %% Save model
torch.save(Model.state_dict(), "video2image_model.pth")

# %% Basic performance measures on training set
Model.eval()

eval_loader = RNN_data.create_dataloader(dataset_Path, video_types = ('circular',), resolution=(480, 640), crop_size=(20, 20), batch_size=1, shuffle=True)

# feed through the first batch
batch = eval_loader[1]

video, label = batch  # video: [B, T, 1, 20, 20], label: [B, 1, 20, 20]
video = video.to(device, dtype=torch.float)

output = Model(video)

output = output.detach().cpu().numpy()

#plot the output and label in one figure
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Prediction")
plt.imshow(output[0], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Label")
plt.imshow(label[0, 0].cpu(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# %%
