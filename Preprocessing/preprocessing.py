import os
import numpy as np
from scipy.signal import wiener
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

def apply_fft(data, t):
    # compute fast fourier transform
    fft_result = fft(data, axis=0)

    # compute frequencies
    freq = fftfreq(len(t), d=(t[1]-t[0]))

    return fft_result, freq

def add_zero_padding(data, num_zeroes):
    # add zero padding to time sequence (should be to nearest power of 2)
    # num_zeroes for exp_data should be 2048 - data.shape[0]
    return np.pad(data, ((0, num_zeroes), (0,0)))

def apply_wiener(fft_data):
    # filter frequency domain image using wiener filter
    filtered_fft = []
    for spectrum in fft_data.T:
        magnitude = np.abs(spectrum)
        #phase = np.angle(spectrum)
        filtered_magnitude = wiener(magnitude)
        filtered_fft.append(filtered_magnitude)  #* np.exp(1j*phase))

    return np.array(filtered_fft).T

def standardize(fft_data):
    # standardize the filtered fft data
    fft_magnitudes = np.abs(fft_data)
    mean = np.mean(fft_magnitudes, axis=0)
    std = np.std(fft_magnitudes, axis=0)
    standardized_data = (fft_magnitudes - mean)/std
    return standardized_data

def extract_peak_features(fft_data, freq):
    fft_magnitude = np.abs(fft_data)
    peak_indices = np.argmax(fft_magnitude, axis=0)  # shape: (num_pixels,)
    peak_magnitudes = fft_magnitude[peak_indices, np.arange(fft_magnitude.shape[1])]
    peak_freqs = freq[peak_indices]
    
    features = np.stack((peak_freqs, peak_magnitudes), axis=1)  # shape: (num_pixels, 2)
    return features

def apply_PCA(X , num_components):
     
    #Step-1
    X_centered = X - np.mean(X , axis = 1, keepdims=True)
     
    #Step-2
    cov_mat = np.cov(X_centered , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(X_centered, eigenvector_subset.T)

     
    return X_reduced

def apply_PCA_SVD(X, num_components):
    # Step-1: Mean center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step-2: Compute SVD instead of covariance matrix
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Step-3: Select the top 'num_components' principal components
    eigenvector_subset = Vt[:num_components, :]  # Shape: (num_components, 307200)
    
    # Step-4: Project data onto the new subspace
    X_reduced = np.dot(X_centered, eigenvector_subset.T)  # Shape: (301, num_components)
    
    return X_reduced

