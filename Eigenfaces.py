#CI Project: Face recognition using Eigenfaces
from sklearn import datasets
#Dataset: Face images dataset from AT&T taken from https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
faces = datasets.fetch_olivetti_faces()
#faces.data = ravelled face image, faces.image = original image, faces.target = labels of each image 
print(faces.data.shape, faces.images.shape,faces.target.shape) 

from matplotlib import pyplot as plt
#Plotting samples from the dataset (Note that the same subject has images with and without glasses)
fig = plt.figure(figsize=(8, 8))
plt.title("Sample faces")
j=0
for i in range(100,125):
    ax = fig.add_subplot(5, 5, j+1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=plt.cm.gray)
    j+=1

#Train-test splitting to test the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(faces.data,
        faces.target,test_size=0.25)
print(X_train.shape, X_test.shape)

#Applying PCA for feature extraction (Here, we take 150 components in the PCA model)
from sklearn import decomposition
pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)

#Plotting the "mean face"
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
plt.title("Mean face")
ax.imshow(pca.mean_.reshape(faces.images[0].shape),cmap=plt.cm.gray)

#Each PCA component is composed of the original 4096 features
print(pca.components_.shape)

fig = plt.figure(figsize=(15, 3))
plt.title("Sample PCA Components")
#Plotting the first 20 PCA components out of 150 in total 
#Here 100% variance is taken so as not to compromise on model accuracy
for i in range(20):
    ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape),
              cmap=plt.cm.gray)

#New train and test data after PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)
print(X_test_pca.shape)

#Applying SVM classifier on data 
#Here, we take C=10 and gamma=0.001 (Trial and error)
from sklearn import svm
clf = svm.SVC(C = 10, gamma = 0.001)
clf.fit(X_train_pca, y_train)

#Plotting sample images from test set
import numpy as np
fig = plt.figure(figsize=(8, 8))
plt.title("Sample Test Images")
j=0
for i in range(50,75):
    ax = fig.add_subplot(5, 5, j + 1, xticks=[], yticks=[])
    ax.imshow(X_test[i].reshape(faces.images[0].shape),
              cmap=plt.cm.gray)
    j+=1

from sklearn import metrics
y_pred = clf.predict(X_test_pca)
print('Accuracy: {:.3f}'.format(clf.score(X_test_pca, y_test)))
print(metrics.classification_report(y_test, y_pred))

#Additional: Plotting Confusion Matrix as Heatmap
#Here, the heatmap may not always be 40 X 40 since the test data may not include images of all the 40 subjects
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = metrics.confusion_matrix(y_test, y_pred)
df = pd.DataFrame(array, index = [i for i in range(0,len(array[0]))],
                  columns = [i for i in range(0,len(array[0]))])
plt.figure(figsize = (10,7))
plt.title("Confusion Matrix: Actual Labels vs Predicted Labels")
sn.heatmap(df, xticklabels=1, yticklabels=1, annot=True)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()