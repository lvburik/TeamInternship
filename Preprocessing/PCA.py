from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from InterpolatedData import *
from Labels import *
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import glob
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data_folder="InterpolatedData"
data=glob.glob(os.path.join(data_folder,"*.npy"))
simulation_data=[]

label_folder="Labels"
label=glob.glob(os.path.join(label_folder,"*.npy"))
labels=[]

for f in data:
    file=np.load(f)
    simulation_data.append(file)

for l in label:
    label=np.load(l)
    labels.append(label)



def PCA(X , num_components): #code for implementing PCA, however it runs slowly
    X_meaned = X - np.mean(X , axis = 1, keepdims=True)
    cov_mat = np.cov(X_meaned , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    X_reduced = np.dot(X_meaned, eigenvector_subset.T)

     
    return X_reduced


import numpy as np

def PCA_SVD(X, num_components): #code for implementing PCA that was used, uses SVD for faster computation
    X_meaned = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_meaned, full_matrices=False)
    eigenvector_subset = Vt[:num_components, :]  # Shape: (num_components, 307200)
    X_reduced = np.dot(X_meaned, eigenvector_subset.T)  # Shape: (301, num_components)
    
    return X_reduced



result=PCA_SVD(simulation_data[0],5)
print(result.shape)


