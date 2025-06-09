import os
import torch
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def Train(epochs, model, dataloader, optimizer, criterion, save_best_model = False, save_path = "best_model.pth"):
    device = device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):  # You can change the number of epochs as needed
        running_loss = 0.0
        num_batches = len(dataloader)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if num_batches > 0 and ((batch_idx + 1)% (num_batches // 10)) == 0:
                print(f"Epoch: {epoch + 1}, Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        if save_best_model and epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs("Saved_models", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("Saved_models", save_path))

# %% feed one video through the model
video_id = 'video_square_3'
mask_id = 'mask_square'

def reconstruct_and_plot(video_id, mask_id, hdf5_path, model, savepath=None, num_frames=26, crop_size=(20, 20), output_size=(2, 2)):
    """
    Reconstructs the output for a video using a sliding window and plots it against the ground truth mask.

    Args:
        video_id (str): Name of the video dataset in the HDF5 file.
        mask_id (str): Name of the mask dataset in the HDF5 file.
        hdf5_path (str): Path to the HDF5 file.
        model (torch.nn.Module): Trained PyTorch model.
        savepath (str, optional): Path to save the reconstructed output plot. If None, the plot is shown.
        num_frames (int): Number of frames to use from the video.
        crop_size (tuple): Size of the crop window (height, width).
        output_size (tuple): Size of the model output (height, width).
    """
    import matplotlib.pyplot as plt

    # Load the video and mask from the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        video = f[video_id][()]  # shape: (num_frames, H, W)
        mask = f[mask_id][()]   # shape: (H, W)

    height, width = video.shape[1:3]
    reconstructed = np.zeros((height, width))
    count_map = np.zeros((height, width))

    # Stride is set to output size
    stride_y, stride_x = output_size

    model.eval()
    for y in range(0, height - crop_size[0] + 1, stride_y):
        for x in range(0, width - crop_size[1] + 1, stride_x):
            crop = video[:num_frames, y:y+crop_size[0], x:x+crop_size[1]]
            crop_tensor = torch.tensor(crop, dtype=torch.float32).unsqueeze(0)  # (1, num_frames, crop_h, crop_w)
            with torch.no_grad():
                output = model(crop_tensor)  # (1, output_h, output_w)
            output_np = output.squeeze(0).numpy()
            cy, cx = y + crop_size[0]//2 - output_size[0]//2, x + crop_size[1]//2 - output_size[1]//2
            reconstructed[cy:cy+output_size[0], cx:cx+output_size[1]] += output_np
            count_map[cy:cy+output_size[0], cx:cx+output_size[1]] += 1

    count_map[count_map == 0] = 1
    reconstructed /= count_map

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    #plt.title('Reconstructed Output')
    plt.imshow(reconstructed, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.subplot(1, 2, 2)
    #plt.title('Ground Truth Mask')
    #plt.imshow(mask, cmap='gray')
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


def evaluate(Model, Dataloader):
    """
    Returns average Accuracy, Precision, Recall, F1 Score, and IoU over the dataset.
    """
    Model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in Dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = Model(inputs)
            # If outputs are logits, apply sigmoid or softmax as appropriate
            if outputs.shape == labels.shape:
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
            else:
                preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_labels.append(labels.cpu().numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    iou = jaccard_score(all_labels, all_preds, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")
    return accuracy, precision, recall, f1, iou