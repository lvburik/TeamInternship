# %%
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# %%
class CustomVideoDataset(Dataset):
    def __init__(self, hdf5_path, video_types=('circular'), resolution=(480, 640), crop_size=(20, 20)):
        self.hdf5_path = hdf5_path
        self.crop_size = crop_size
        self.resolution = resolution
        self.video_types = video_types

        # Open HDF5 file to get video indices for each type
        with h5py.File(self.hdf5_path, 'r') as f:
            self.video_keys = []
            for vtype in self.video_types:
                self.video_keys += [key for key in f.keys() if key.startswith(f'video_{vtype}_')]
        self.number_of_videos = len(self.video_keys)

        

        # All combinations of flip, mirror, and rotate (0, 90, 180, 270)
        self.augmentations = [
            (False, False, 0), (True, False, 0), (False, True, 0), (True, True, 0),
            (False, False, 90), (True, False, 90), (False, True, 90), (True, True, 90)
        ]

        # Load all masks for each video_type
        self.masks = {}
        with h5py.File(self.hdf5_path, 'r') as f:
            for vtype in self.video_types:
                mask_key = f'mask_{vtype}'
                if mask_key in f:
                    self.masks[vtype] = np.array(f[mask_key])
                else:
                    raise KeyError(f"Mask {mask_key} not found in HDF5 file.")
                
        self.num_videos_per_type = {vtype: sum(1 for key in self.video_keys if key.split('_')[1] == vtype) for vtype in self.video_types}
        self.crops_dict = self.find_crops()
        self.typecutoffs = np.cumsum([len(self.crops_dict[vtype])*self.num_videos_per_type[vtype]*len(self.augmentations) for vtype in self.video_types])
        

    def __len__(self):
        # sum the length of all coordinate lists in the crops_dict
        length = 0
        for vtype in self.video_types:
            if vtype in self.crops_dict:
                length += len(self.crops_dict[vtype]) * len(self.augmentations) * self.num_videos_per_type[vtype]

        return length

    def __getitem__(self, index):
        # Determine which video type this index belongs to
        video_type_idx = np.searchsorted(self.typecutoffs, index, side='right')
        vtype = self.video_types[video_type_idx]

        # Calculate the base index within this video type
        if video_type_idx == 0:
            base_index = index
        else:
            base_index = index - self.typecutoffs[video_type_idx - 1]

        # Get number of crops and videos for this type
        num_crops = len(self.crops_dict[vtype])
        num_videos = self.num_videos_per_type[vtype]
        num_augs = len(self.augmentations)

        # Calculate which video, crop, and augmentation this index refers to
        video_idx = base_index // (num_crops * num_augs)
        rem = base_index % (num_crops * num_augs)
        crop_idx = rem // num_augs
        aug_idx = rem % num_augs

        crop_top_left = self.crops_dict[vtype][crop_idx]
        flip, mirror, rotate = self.augmentations[aug_idx]
        
        video_key = self.video_keys[video_idx]
        # Extract video_type from key: video_{video_type}_idx
        vtype = video_key.split('_')[1]

        with h5py.File(self.hdf5_path, 'r') as f:
            video = f[video_key]
            cropped_video = video[:, crop_top_left[0]:(crop_top_left[0] + self.crop_size[0]),
                                    crop_top_left[1]:(crop_top_left[1] + self.crop_size[1])]
            assert cropped_video.shape[1:] == self.crop_size, f"cropped_video shape {cropped_video.shape} is invalid"

        label_data = self.masks[vtype][crop_top_left[0]:(crop_top_left[0] + self.crop_size[0]),
                                       crop_top_left[1]:(crop_top_left[1] + self.crop_size[1])]
        assert label_data.shape == self.crop_size, f"label_data shape {label_data.shape} is invalid"

        if flip:
            cropped_video = np.flip(cropped_video, axis=2)
            label_data = np.flip(label_data, axis=1)

        if mirror:
            cropped_video = np.flip(cropped_video, axis=1)
            label_data = np.flip(label_data, axis=0)

        if rotate != 0:
            k = rotate // 90
            cropped_video = np.rot90(cropped_video, k=k, axes=(1, 2))
            label_data = np.rot90(label_data, k=k, axes=(0, 1))

        return torch.tensor(cropped_video.copy(), dtype=torch.float32), torch.tensor(label_data.copy(), dtype=torch.float32)
    
    def find_crops(self):
        # Find all crops containing at least 5% non-zero pixels in the mask of each video type
        crop_dict = {}
        crop_h, crop_w = self.crop_size
        mask_h, mask_w = self.resolution

        crop_rows = mask_h - crop_h
        crop_cols = mask_w - crop_w
        

        for vtype, mask in self.masks.items():
            coords = []
            for row in range(crop_rows):
                for col in range(crop_cols):
                    top = row
                    left = col
                    crop = mask[top:top + crop_h, left:left + crop_w]
                    if crop.size == 0:
                        continue
                    nonzero_ratio = np.count_nonzero(crop) / crop.size
                    if nonzero_ratio >= 0.05:
                        coords.append((row, col))
            crop_dict[vtype] = coords
        return crop_dict
        
# %% create a dataloader
from torch.utils.data import DataLoader

def create_dataloader(hdf5_path, video_types = 'circular', resolution=(480, 640), crop_size=(20, 20), batch_size=32, shuffle=True, crop_fraction=0.001):
    dataset = CustomVideoDataset(hdf5_path, video_types, resolution, crop_size)
    # Calculate number of samples to use (1% of total)
    num_samples = max(1, int(len(dataset) * crop_fraction))
    # Randomly select indices for this epoch
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader

# %%
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Example usage
    hdf5_path = 'compacted_improved.h5'  # Path to your HDF5 file
    label_path = 'label.png'  # Path to your label image
    video_types = ('circular',)  # Types of videos to include
    resolution = (480, 640)  # Original video resolution
    crop_size = (20, 20)  # Size of the crops

    dataloader = create_dataloader(hdf5_path, video_types, resolution, crop_size, batch_size=32)
    dataset = dataloader.dataset
    print(dataset[10])

    



# %%
