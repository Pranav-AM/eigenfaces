# Facial Recognition using Eigenfaces

The objective of this assignment is to correctly classify the given
images into one of 40 labels. 

The dataset used here was taken from the sklearn datasets found in the url:
https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.htm,
which contains 400 face images of size 64 x 64 pixels corresponding
to 40 given labels (each label has 10 images).

During the preprocessing stage, the dataset was first split into
training and test sets using 25% split. PCA was then applied to
this data for feature extraction. 150 PCA components, each having
4096 features, were taken and a new training and test set was
made using the 150 PCA components. The preprocessed dataset
was then used as input for SVM (Support Vector Machine) for
classification.

It was observed that the SVM after applying PCA gave an
accuracy of 0.92, precision of 0.97, recall of 0.92 and f1-score of
0.93. To visualize this better, a confusion matrix was plotted (of the
actual labels vs the predicted labels).
