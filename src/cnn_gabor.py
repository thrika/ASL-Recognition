import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import os, re
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.datasets import imdb

######
path = sorted(os.listdir('./gabor_features'))
i = 0
feature_array = []
labels_array = []
for f in path:
    if re.match('g', f):
        features = np.load('./gabor_features/' + f).tolist()
        feature_array += features
        i += 1

feature_array = np.array(feature_array)

for f in path:
    if re.match('l', f):
        labels = np.load('./gabor_features/' + f).tolist()
        labels_array += labels
        i += 1

data = np.array(feature_array)
labels = np.array(labels_array)

# Performing PCA with dimensionality reduction
pca = PCA()
pca = pca.fit_transform(data)
print("Shape of Training data PCA: ", pca.shape)

# x_train, x_test, y_train, y_test = train_test_split(pca, labels, test_size=0.3, random_state=0)

model = SVC(gamma='auto').fit(data, labels)
# y_pred = model.predict(x_test)

# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

# Split the data to test and train and performing SVM
x_train, x_test, y_train, y_test = train_test_split(pca, labels, test_size=0.3, random_state=0)

model = MLPClassifier(hidden_layer_sizes=(40,40,40,40), max_iter=40000)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

dump(model, 'gabormlp1.joblib')