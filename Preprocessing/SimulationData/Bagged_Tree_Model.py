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

training_data=simulation_data[:12]
test_data=simulation_data[-9:]
training_labels=labels[:12]
test_labels=labels[-9:]

print(len(simulation_data[0][0]))


filtered_data=gaussian_filter(training_data,2)
t = np.linspace(0.5, 301*0.5, 301)
log_normalized_data=np.log(filtered_data)
log_t=np.log(t)
first_derivative=[]
dev_1=[]

for i in range(len(training_data)):
    for j in range(len(training_data[0][0])):
        der=np.gradient(log_normalized_data[i][:,j],t)
        first_derivative.append(der)
    dev_1.append(np.array(first_derivative))
    first_derivative=[]

for i in range(len(dev_1)):
    dev_1[i]=dev_1[i].T

print(len(dev_1[0][0]))

def ensemble(X,y, test_data):
    clf = BaggingClassifier(estimator=DecisionTreeClassifier(), max_samples=1.0, oob_score=True,
                        n_estimators=10, random_state=13).fit(X, y)
    return clf.predict(test_data)


result= ensemble(first_derivative, training_labels, test_data)
mse=np.sum(result[0]-test_labels[0])^2

print(mse)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(training_data)
x_test=scaler.transform(test_data)

model = LogisticRegression()
model.fit(x_train_scaled, labels)

y_pred = model.predict(x_test)



# Step 2: Apply PCA
#num_components = 5  # Adjust based on explained variance
#pca = PCA(n_components=num_components)
#tsr_pca = pca.fit_transform(tsr_scaled)

# Step 3: Reconstruct the TSR data using selected principal components
#tsr_reconstructed = pca.inverse_transform(tsr_pca)
