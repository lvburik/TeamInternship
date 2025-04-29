# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import os
from PIL import Image
from sklearn.cluster import KMeans



# %% code for detection
Folder_Directory = "C:/Users/20202555/OneDrive - TU Eindhoven/Documents/M AI&ES - Year 1/Team-internship/Experimental_Data"

File_list = os.listdir(Folder_Directory)

Video_ID = 1

# %%
Video = np.load(Folder_Directory+"/"+File_list[Video_ID])

# %%
(N_frames, N_pixels) = np.shape(Video)

Deviation_Video = Video - np.mean(Video, axis = 1)
# %% perform fourier transform over full video
fft_Deviation_Video = fft(Deviation_Video, axis = 0)

# %%
fft_Deviation_Video_Magnitudes = np.abs(fft_Deviation_Video)

# %%
Frame = [np.max(Deviation_Video, axis = 0), np.min(Deviation_Video, axis = 0), np.median(Deviation_Video, axis = 0), np.std(Deviation_Video, axis = 0)]
frame = np.max(fft_Deviation_Video_Magnitudes[2:30, :], axis = 0)
# 1 noise, 1 damage, and 1 sample cluster
clustering = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(np.transpose(fft_Deviation_Video_Magnitudes[1:500, :]))

labels = clustering.labels_

# %%
clustering2 = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(np.transpose(Frame))
labels2 = clustering2.labels_

# %% load the mask
Mask_Directory = Folder_Directory+"/"+File_list[0]

im_mask = Image.open(Mask_Directory)
Mask = np.array(im_mask.getdata())


group_a = frame[Mask == 1]
group_b = frame[Mask == 0]


# %% calculate the confusion matrix for this sample: (including column for outside of sample)
Confusion_matrix = np.zeros((2, 3))
for Pixel in range(N_pixels):
    Confusion_matrix[Mask[Pixel]][labels[Pixel]] += 1

print(Confusion_matrix)




# %% Plot a few images (results)
Mask_IM = Mask.reshape((480, 640))
plt.imshow(Mask_IM, cmap='gray')
plt.title('mask')
              
# %%
Predictions_IM = labels.reshape((480, 640))/2
plt.imshow(Predictions_IM, cmap='gray')
plt.title('predictions using K-means on fft data')
# %%
Predictions_IM2 = labels2.reshape((480, 640))/2
plt.imshow(Predictions_IM2, cmap = 'gray')
plt.title('predictions using K-means on features of raw data')

# %%
Video_frame_IM = Video[1300, :].reshape((480, 640))
plt.imshow(Video_frame_IM, cmap='gray')
plt.title("raw video frame")


# %%
Frame_IM = frame.reshape((480, 640))
plt.imshow(Frame_IM, cmap='gray')
plt.title("maximum of fft of data")
# %%
