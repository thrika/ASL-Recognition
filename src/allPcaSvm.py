import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
import os, re

######
path = sorted(os.listdir('./gabor_features_test'))
i=0
feature_array=[]
labels_array=[]
for f in path:
   if re.match('g', f):
       features = np.load('./gabor_features_test/'+f).tolist()
       feature_array += features
       i+=1
       
feature_array = np.array(feature_array)

for f in path:
   if re.match('l', f):
       labels = np.load('./gabor_features_test/'+f).tolist()
       labels_array += labels
       i+=1
       
data = np.array(feature_array)
labels=np.array(labels_array)

#Performing PCA with dimensionality reduction
pca = PCA()
pca = pca.fit_transform(data)
print("Shape of Training data PCA: ", pca.shape)

#x_train, x_test, y_train, y_test = train_test_split(pca, labels, test_size=0.3, random_state=0)

model = SVC(gamma='auto').fit(data, labels)
#y_pred = model.predict(x_test)

#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#Split the data to test and train and performing SVM
x_train, x_test, y_train, y_test = train_test_split(pca, labels, test_size=0.3, random_state=0)
model = SVC(gamma='auto').fit(x_train, y_train)
y_pred = model.predict(x_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Training: ", model.score(x_train, y_train))
print("Testing: ", model.score(x_test, y_test))

#Computing the SVM for the complete dataset
#dump(SVC(gamma='auto').fit(pca, labels), 'svm_with_gf_with_pca.joblib')
#dump(model, 'finalpcasvm.joblib')