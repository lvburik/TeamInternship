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



def PCA(X , num_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 1, keepdims=True)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(X_meaned, eigenvector_subset.T)

     
    return X_reduced


import numpy as np

def PCA_SVD(X, num_components):
    # Step-1: Mean center the data
    X_meaned = X - np.mean(X, axis=0)
    
    # Step-2: Compute SVD instead of covariance matrix
    U, S, Vt = np.linalg.svd(X_meaned, full_matrices=False)
    
    # Step-3: Select the top 'num_components' principal components
    eigenvector_subset = Vt[:num_components, :]  # Shape: (num_components, 307200)
    
    # Step-4: Project data onto the new subspace
    X_reduced = np.dot(X_meaned, eigenvector_subset.T)  # Shape: (301, num_components)
    
    return X_reduced



result=PCA_SVD(simulation_data[0],5)
print(result.shape)


