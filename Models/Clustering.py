# %%# Clustering.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from sklearn.decomposition import PCA
import os
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter

def Plot_Clustering(Video_Directory, Video_ID = 1, n_clusters = 2, threshold = 0.3,FFT = True, TSR = None, apply_moving_average = False):
    """ Plots the clustering of a video based on the specified method. """
    File_list = os.listdir(Video_Directory)
    Video = np.load(Video_Directory + "/" + File_list[Video_ID])


    num_frames = Video.shape[0]
    # Reshape video to (num_frames, height, width)
    Video = Video.reshape((num_frames, 480, 640))
    # Define crop limits (y_start, y_end), (x_start, x_end)
    crop_limits = [(0, 480), (0, 640)]
    y_start, y_end = crop_limits[0]
    x_start, x_end = crop_limits[1]
    # Crop video using crop_limits
    Video = Video[:, y_start:y_end, x_start:x_end]

    # reshape to (num_frames, 300*400)
    num_frames = Video.shape[0]
    Video = Video.reshape(num_frames, (x_end - x_start) * (y_end-y_start))
    
    # Perform clustering
    labels = K_means_predict(Video, n_clusters, FFT = FFT, TSR = TSR).reshape(((y_end-y_start), (x_end - x_start)))
    
    if apply_moving_average:
        #apply moving average to the labels image
        # Apply moving average (uniform filter) with filter size 10 and stride 1
        labels = uniform_filter(labels.astype(float), size=10, mode='nearest')

        
    

    plt.figure(figsize=(10, 5))
    plt.imshow(labels < threshold, aspect='auto', cmap='gray')
    string = ''
    if FFT:
        string += 'FFT '
    if FFT and TSR != None:
        string += 'and '
    if TSR != 0:
        if TSR == 1:
            string += '1st order TST'
        if TSR == 2:
            string += '2nd order TSR'
    #plt.title(f'Clustering using '+string)
    
    plt.xticks([])  # Remove x-axis numbers
    plt.yticks([])  # Remove y-axis numbers
    plt.show()

    return labels

def K_means_predict(Video, n_clusters = 2, FFT = True, TSR = None):
    
    
        

    if TSR != None:
        Video = np.log10(Video)
        for ii in range(TSR):
            Video = np.diff(Video, axis=0)


    if FFT:
        Video = Video - np.mean(Video, axis=0)
        # Apply FFT to the video data
        Video = fft(Video, axis=0)[:100][:]
        Video = np.abs(Video)  # Use magnitudes of the FFT coefficients
       
    
    Preprocessed = Extract_features(Video)
    

    # Perform K-means clustering on each pixel
    
    clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(Preprocessed)
    
    return clustering.labels_


def Extract_features(Video):
    """ Extracts the minimum, maximum, median, standerd deviation, and mean of each of the video pixels. """
    features = np.zeros((Video.shape[1], 5))  # 5 features per pixel
    features[:, 0] = np.max(Video, axis=0)  # max
    features[:, 1] = np.min(Video, axis=0)  # min
    features[:, 2] = np.median(Video, axis=0)  # median
    features[:, 3] = np.std(Video, axis=0)  # std
    features[:, 4] = np.mean(Video, axis=0)  # mean
    
    return features


# %%

Video_Directory = 'C:/Users/20202555/OneDrive - TU Eindhoven/Documents/M AI&ES - Year 1/Team-internship/Experimental_Data/Test'
    
   
    
# %%
Plot_Clustering(Video_Directory, Video_ID=1, n_clusters=3, FFT = True, TSR = 2)
Plot_Clustering(Video_Directory, Video_ID=1, n_clusters=3, FFT = True, TSR = 1)
Plot_Clustering(Video_Directory, Video_ID=1, n_clusters=3, FFT = True, TSR = None)
Plot_Clustering(Video_Directory, Video_ID=1, n_clusters=3, FFT = False, TSR = 2)
Plot_Clustering(Video_Directory, Video_ID=1, n_clusters=3, FFT = False, TSR = 1)
Plot_Clustering(Video_Directory, Video_ID=1, n_clusters=3, FFT = False, TSR = None)


# %%

# %%
