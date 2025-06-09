# %% import necessary libraries
import models_dataset_dataloader
import Training_and_evaluation

import torch
import torch.nn as nn
import torch.optim as optim

# %% define a simple linear model
class linearmodel(nn.Module):
    def __init__(self, input_frames=26, input_size=(100, 100), hidden_sizes=(256, 128, 64), output_size=(2, 2)):
        super(linearmodel, self).__init__()
        self.input_frames = input_frames
        in_features = input_frames * input_size[0] * input_size[1]
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, output_size[0] * output_size[1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 26, 20, 20)
        x = x.view(x.size(0), -1)  # Flatten to (B, 26*20*20)
        out = self.model(x)
        out = out.view(x.size(0), 2, 2)
        return out
    
# %% define the dataset and dataloader
if __name__ == "__main__":
    hdf5_path = 'Datasets/'+'train2%.h5'  # Replace with your actual HDF5 file path
    dataloader = models_dataset_dataloader.create_dataloader(
        hdf5_path, video_types=('circular', 'rec', 'square', 'triangular'),	
        resolution=(480, 640), crop_size=(100, 100), label_size=(2, 2))
    
    

# %% create the model
if __name__ == "__main__":
    Model = linearmodel()

# %% define a loss function and optimizer
if __name__ == "__main__":
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Model.parameters(), lr=0.001)

# %% train the model
if __name__ == "__main__":
    Training_and_evaluation.Train(10,
        Model, dataloader, optimizer, criterion,
        save_best_model=True, save_path="linear_model100_100.pth"
    )

# %% set the model to evaluation mode
if __name__ == "__main__":  
    Model.eval()

# %% load the best model
if __name__ == "__main__":
    Model.load_state_dict(torch.load("Saved_Models/linear_model100_100.pth"))

# %% load the test dataset
if __name__ == "__main__":
    hdf5_path = 'Datasets/'+'test2%.h5'  # Replace with your actual HDF5 file path
    dataloader = models_dataset_dataloader.create_dataloader(
        hdf5_path, video_types=('circular', 'rec', 'square', 'triangular'),
        resolution=(480, 640), crop_size=(100, 100), label_size=(2, 2))
    


# %% feed one video through the model
if __name__== "__main__":
    video_id = 'video_circular_0'
    mask_id = 'mask_' + video_id.split('_')[1]  # Extract mask_id from video_id

    Training_and_evaluation.reconstruct_and_plot(
        video_id, mask_id, hdf5_path, Model, num_frames=Model.input_frames, crop_size=(100, 100))
# %%
