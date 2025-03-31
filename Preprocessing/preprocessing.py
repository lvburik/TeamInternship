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

def standardize(filtered_fft_data):
    # standardize the filtered fft data
    fft_magnitudes = np.abs(filtered_fft_data)
    mean = np.mean(fft_magnitudes, axis=0)
    std = np.std(fft_magnitudes, axis=0)
    standardized_data = (fft_magnitudes - mean)/std
    return standardized_data
