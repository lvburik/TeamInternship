import numpy as np
from scipy.fftpack import fft, fftfreq

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

def standardize(data):
    # standardize fft data
    fft_magnitudes = np.abs(data)
    mean = np.mean(fft_magnitudes, axis=0)
    std = np.std(fft_magnitudes, axis=0)
    standardized_data = (fft_magnitudes - mean)/std
    return standardized_data

def extract_patches(data, mask, patch_size=16):
    # reshape into (480, 640, num_freqs)
    data = data.T.reshape(480, 640, data.shape[0])

    patches = []
    labels = []

    # extract non-overlapping patches and their labels
    for i in range(0, 480, patch_size):
        for j in range(0, 640, patch_size):
            if i + patch_size <= 480 and j + patch_size <= 640:
                patch = data[i:i+patch_size, j:j+patch_size]
                # take mean magnitude for each frequency bin of pixels in patch
                patch = np.mean(patch, axis=(0,1))

                label = mask[i:i+patch_size, j:j+patch_size]
                # label patch according to patch mean
                label = 1 if np.mean(label) > 0.5 else 0

                patches.append(patch)
                labels.append(label)

    return np.array(patches), np.array(labels)

def extract_cnn_patches(data, mask, patch_size=128, overlap=0, neg_patch_prob=1.0, augment=False):
    # reshape into (480, 640, num_freqs)
    height, width = 480, 640
    data = data.T.reshape(height, width, data.shape[0])

    stride = int(patch_size - overlap * patch_size)

    patches = []
    labels = []
    
    # extract overlapping patches and their labels
    for i in range(0, height-patch_size+1, stride):
        for j in range(0, width-patch_size+1, stride):
            
            patch = data[i:i+patch_size, j:j+patch_size]
            patch_mask = mask[i:i+patch_size, j:j+patch_size]
            
            if patch_mask.sum() / (patch_size * patch_size) > 0.05:
                # prioritize patches with defects
                patches.append(patch)
                labels.append(patch_mask)
                
                if augment:
                    # apply augmentation
                    for _ in range(np.random.randint(1, 4)):
                        patch, patch_mask = augment_patch(patch, patch_mask)
                        patches.append(patch)
                        labels.append(patch_mask)
            elif np.random.random() < neg_patch_prob:
                # randomly include some patches with no defects
                patches.append(patch)
                labels.append(patch_mask)

    # handle bottom edge
    if (height - patch_size) % stride != 0:
        i = height - patch_size
        for j in range(0, width - patch_size + 1, stride):
            patch = data[i:i + patch_size, j:j + patch_size]
            patch_mask = mask[i:i + patch_size, j:j + patch_size]
            if patch_mask.sum() > 0 or np.random.random() < neg_patch_prob:
                patches.append(patch)
                labels.append(patch_mask)

    # handle right edge
    if (width - patch_size) % stride != 0:
        j = width - patch_size
        for i in range(0, height - patch_size + 1, stride):
            patch = data[i:i + patch_size, j:j + patch_size]
            patch_mask = mask[i:i + patch_size, j:j + patch_size]
            if patch_mask.sum() > 0 or np.random.random() < neg_patch_prob:
                patches.append(patch)
                labels.append(patch_mask)

    # bottom-right corner
    i, j = height - patch_size, width - patch_size
    patch = data[i:i + patch_size, j:j + patch_size]
    patch_mask = mask[i:i + patch_size, j:j + patch_size]
    if patch_mask.sum() > 0 or np.random.random() < neg_patch_prob:
        patches.append(patch)
        labels.append(patch_mask)

    return np.array(patches), np.array(labels)

def apply_PCA_SVD(X, num_components):
    # mean center the data
    X_centered = X - np.mean(X, axis=0)
    
    # compute SVD instead of covariance matrix
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # select the top 'num_components' principal components
    eigenvector_subset = Vt[:num_components, :]  # shape: (num_components, 307200)
    
    # project data onto the new subspace
    X_reduced = np.dot(X_centered, eigenvector_subset.T)
    
    return X_reduced

def apply_PCA_patches(patches, num_components):
    P, H, W, F = patches.shape

    # reshape to (P*H*W, F)
    flat = patches.reshape(-1, F)

    # apply your PCA function
    reduced_flat = apply_PCA_SVD(flat, num_components)  # shape: (P*H*W, num_components)

    # reshape back to (, H, W, num_components)
    reduced = reduced_flat.reshape(P, H, W, num_components)
    return reduced

def augment_patch(patch, mask):
    # apply random horizontal flip
    if np.random.random() < 0.5:
        patch = np.flip(patch, axis=1)
        mask = np.flip(mask, axis=1)

    # apply random vertical flip    
    if np.random.random() < 0.5:
        patch = np.flip(patch, axis=0)
        mask = np.flip(mask, axis=0)
    # apply random rotation
    k = np.random.choice([0, 1, 2, 3])
    patch = np.rot90(patch, k=k, axes=(0, 1))
    mask = np.rot90(mask, k=k, axes=(0, 1))

    return patch, mask