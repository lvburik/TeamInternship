import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import piq
from thermal_dataset import ThermalDataset
import numpy as np
from matplotlib import pyplot as plt
from train_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(doubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class SmallerUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SmallerUNet, self).__init__()
        self.down1 = doubleConv(in_channels, 32)
        self.down2 = doubleConv(32, 64)
        
        self.bottleneck = doubleConv(64, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = doubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = doubleConv(64, 32)       
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # encoding
        enc1 = self.down1(x)
        enc2 = self.down2(nn.MaxPool2d(2)(enc1))

        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc2))

        # decoding
        dec2 = self.dec2(torch.cat([self.up2(bottleneck), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], 1)) 

        return self.final_conv(dec1)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.down1 = doubleConv(in_channels, 64)
        self.down2 = doubleConv(64, 128)
        self.down3 = doubleConv(128, 256)
        self.down4 = doubleConv(256, 512)
        
        self.bottleneck = doubleConv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = doubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = doubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = doubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = doubleConv(128, 64)       
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # encoding
        enc1 = self.down1(x)
        enc2 = self.down2(nn.MaxPool2d(2)(enc1))
        enc3 = self.down3(nn.MaxPool2d(2)(enc2))
        enc4 = self.down4(nn.MaxPool2d(2)(enc3))
        
        # bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))

        # decoding
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], 1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], 1))
        
        return self.final_conv(dec1)

class Network(torch.nn.Module): 
    def __init__(self, n_in, n_classes=1): 
        super(Network,self).__init__() 

        # convolutional and pooling blocks
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.upsample1 = torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=4, padding=1, output_padding=3)
        self.upsample2 = torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = torch.nn.Conv2d(16, n_classes, kernel_size=1)

    
    def forward(self,x): 
        x = self.block1(x)
        x = self.block2(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.final_conv(x)

        # get output to exact original size
        #x = F.interpolate(x, size=(480, 640), mode='bilinear', align_corners=False)

        return x 

def combined_loss(preds, targets, pos_weight, alpha=0.7):
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce = bce_loss_fn(preds, targets)
    preds_prob = torch.sigmoid(preds)
    ssim_val = piq.ssim(preds_prob, targets, data_range=1.0)
    return alpha * bce + (1 - alpha) * (1 - ssim_val)

def train_model(train_loader, num_epochs=10): 
    modelname = "smallunet_pca_200"
    #model = Network(n_in=10).to(device)
    #model = UNet(in_channels=10).to(device)
    model = SmallerUNet(in_channels=103).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.5]).to(device)) 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            
            # format for model
            # X_batch = [num_patches, num_freqs, patch_size, patch_size]
            X_batch = X_batch.squeeze(0)
            X_batch = X_batch.permute(0, 3, 2, 1)

            # y_batch = [num_patches, 1, patch_size, patch_size]
            y_batch = y_batch.squeeze(0).unsqueeze(1).float()

            # move to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # obtain predictions
            y_pred = model(X_batch)

            # calculate loss, backprop
            optimizer.zero_grad()
            loss = combined_loss(y_pred, y_batch, pos_weight=torch.tensor([3.5]).to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"[{epoch + 1}] epoch loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"{modelname}.pth")
    print(f"{modelname} saved")

    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss")

    plt.savefig(f"training_loss_{modelname}.png")
    plt.close()

    return model

def test_model(model, test_loader, threshold=0.4):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # format for model
            X_batch = X_batch.squeeze(0)
            X_batch = X_batch.permute(0, 3, 2, 1)

            y_batch = y_batch.squeeze(0).unsqueeze(1).float()
            
            # move to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # obtain predictions
            y_batch_pred = model(X_batch)
            predicted = (torch.sigmoid(y_batch_pred) > threshold).float() 
      
            # evaluate model
            y_batch_true = y_batch.view(-1).cpu().numpy().astype(int) 
            y_batch_pred = predicted.view(-1).cpu().numpy().astype(int)

            y_true.append(y_batch_true)
            y_pred.append(y_batch_pred)
            break
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    evaluate(y_true, y_pred)

def main(sim_data_path, exp_data_path):

    # initialize training dataset
    train_dataset = ThermalDataset(
        file_paths=TRAIN_DATA,
        data_dir=exp_data_path, 
        mask_map=MASK_MAP,
        apply_PCA=False,
        extract_patches=False,
        extract_cnn_patches=True,
    )
    
    # initialize testing dataset
    test_dataset = ThermalDataset(
        file_paths=TEST_DATA,
        data_dir=exp_data_path,
        mask_map=MASK_MAP,
        apply_PCA=False,
        extract_patches=False,
        extract_cnn_patches=True,
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

      # train model
    print("training model")
    model = train_model(train_dataloader, num_epochs=1)
    #model = UNet(in_channels=10)
    #model = Network(n_in=103)
    #model.load_state_dict(torch.load("./cnn_model.pth"))
    
    # test model
    print("testing model")
    test_model(model, test_dataloader, threshold=0.35)

if __name__ == "__main__":
    # read file path for data
    sim_data_path = sys.argv[1]
    exp_data_path = sys.argv[2]
    main(sim_data_path, exp_data_path)