# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=26, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # x: (B, 13, 20, 20)
        c1 = self.down1(x)         # (B, 32, 20, 20)
        p1 = self.pool1(c1)        # (B, 32, 10, 10)
        c2 = self.down2(p1)        # (B, 64, 10, 10)
        p2 = self.pool2(c2)        # (B, 64, 5, 5)

        bn = self.bottleneck(p2)   # (B, 128, 5, 5)

        u2 = self.up2(bn)          # (B, 64, 10, 10)
        u2 = torch.cat([u2, c2], dim=1)  # (B, 128, 10, 10)
        c3 = self.conv2(u2)        # (B, 64, 10, 10)

        u1 = self.up1(c3)          # (B, 32, 20, 20)
        u1 = torch.cat([u1, c1], dim=1)  # (B, 64, 20, 20)
        c4 = self.conv1(u1)        # (B, 32, 20, 20)

        out = self.final(c4)       # (B, 1, 20, 20)
        return out

# %% define dataloader
import RNN_data

dataloader = RNN_data.create_dataloader(
    'compacted_improved.h5',
    video_types=('circular', 'square', 'rec'),
    resolution=(480, 640),
    crop_size=(20, 20),
    batch_size=32,
    shuffle=True,
    crop_fraction=0.001
)

# %% define model and submodels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=26, out_channels=1).to(device)
# %% define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %% training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        num_batches = len(dataloader)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
# %%
# %% Basic performance measures on training set
model.eval()

eval_loader = RNN_data.create_dataloader('compacted_improved.h5', video_types = ('circular',), resolution=(480, 640), crop_size=(20, 20), batch_size=1, shuffle=True)

# feed through the first batch
for batch in eval_loader:
    video, label = batch  # video: [B, T, 1, 20, 20], label: [B, 1, 20, 20]
    video = video.to(device, dtype=torch.float)
    output = model(video)

    output = output.detach().cpu().numpy()
    break


# %%


# %%plot the output and label in one figure
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Prediction")
plt.imshow(output[0, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Label")
plt.imshow(label[0].cpu(), cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# %%
