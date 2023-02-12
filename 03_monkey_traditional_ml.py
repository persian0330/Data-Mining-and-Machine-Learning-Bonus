"""
Created on June 22, 2021
@author: J. Czech
Machine Learning Group, TU Darmstadt
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


def get_train_val_data(data_dir: str, shape: tuple) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Reads the training and validation image data from the given data directory and
     scales all images to the given to the input shape.
     The target vectors (y_train, y_val) should contain the class labels as integer values."""

    import glob
    train_folder = os.path.join(data_dir,"training/training")
    x = []
    y = []

    for i in range(10):
        # for filename in glob.glob(train_folder + "/n{}/*.jpg".format(i)):
        for filename in glob.glob(train_folder + "/n{}/*".format(i)):
            image = cv2.imread(filename)
            image = cv2.resize(image, shape) # resize the image
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # change to RGB
            x.append(image_RGB)
            y.append(i)
    X_train = np.array(x)
    y_train = np.array(y)

    # print("Training Dataset:")
    # print(X_train.shape)
    # print(y_train.shape)

    test_folder = os.path.join(data_dir,"validation/validation")
    x = []
    y = []

    for i in range(10):
        for filename in glob.glob(test_folder + "/n{}/*".format(i)):
            image = cv2.imread(filename)
            image = cv2.resize(image, shape) # resize the image
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # change to RGB
            x.append(image_RGB)
            y.append(i)
    X_val = np.array(x)
    y_val = np.array(y)


    # print("Validation Dataset:")
    # print(X_val.shape)
    # print(y_val.shape)
    return X_train, y_train, X_val, y_val

def plot_pca_data(X_data_pca: np.ndarray, y_data: np.ndarray, filename: str):
    """Plots the first two principal components in a 2d scatter plot and saves the figure."""
    x1 = X_data_pca[:, 0]
    x2 = X_data_pca[:, 1]
    y = y_data

    plt.scatter(x1, x2, c=y)
    plt.title(filename)
    plt.xlabel('the first principal component in X_data_pca')
    plt.ylabel('the second principal component in X_data_pca')
    plt.show()



def get_histogram_data(X_data, nb_bins, color_channels=3) -> np.ndarray:
    """Extracts a histogram for each channel and returns the features """
    x=[]

    for i in range(X_data.shape[0]):
        img = X_data[i,:,:,:]
        hist_r = cv2.calcHist([img], [0], None, [nb_bins], [0, 255])
        hist_g = cv2.calcHist([img], [1], None, [nb_bins], [0, 255])
        hist_b = cv2.calcHist([img], [2], None, [nb_bins], [0, 255])
        hist_end = (np.concatenate([hist_r, hist_g, hist_b])).T
        x.append(hist_end)
    X_hist_basis = np.array(x)
    X_hist = X_hist_basis.squeeze()
    return X_hist



def evaluate_knn(X_train_pca, y_train, X_val_pca, y_val, k: int) -> (float, float):
    """Evaluates a k-nearest-Neighbour classifier with value k and returns the training and validation accuracy."""
    classifier = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    classifier.fit(X_train_pca, y_train)
    train_acc = classifier.score(X_train_pca, y_train)
    val_acc = classifier.score(X_val_pca, y_val)

    return  train_acc, val_acc



def apply_pca(X_train: np.ndarray, X_val: np.ndarray, n_components: int) -> (np.ndarray, np.ndarray):
    """Returns the transformed data for X_train and X_val with n_components as principal components."""
    # if len(X_train.shape)>2:
    #     X_train = (X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
    #
    # if len(X_val.shape)>2:
    #     X_val = (X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2] * X_val.shape[3]))

    X_train_reshape = X_train.reshape(X_train.shape[0],-1)
    X_train = X_train_reshape - np.mean(X_train_reshape, axis= 0).reshape(1,-1)
    X_val_reshape = X_val.reshape(X_val.shape[0],-1)
    X_val = X_val_reshape - np.mean(X_val_reshape,axis = 0).reshape(1,-1)



    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    return X_train_pca, X_val_pca


def train_custom_classifier(X_train, X_train_pca, X_train_pca_hist,
                            X_val, X_val_pca, X_val_pca_hist,
                            y_train, y_val) -> (float, float):
    """Trains a custom non neural network classifier and returns the training and validation accuracy."""


    X_train_forest = np.hstack((X_train_pca, X_train_pca_hist))
    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train_forest, y_train)
    train_acc = rfc.score(X_train_forest, y_train)
    X_val_forest = np.hstack((X_val_pca, X_val_pca_hist))
    val_acc = rfc.score(X_val_forest, y_val)
    return train_acc, val_acc

def main():
    # for reproducibility
    np.random.seed(42)

    data_dir = './data/kaggle/10-monkey-species'
    print('Reading image data...')
    get_train_val_data(data_dir, (224, 224))
    X_train, y_train, X_val, y_val = get_train_val_data(data_dir, (224, 224))

    print('Apply PCA on original data...')
    X_train_pca, X_val_pca = apply_pca(X_train, X_val, 2)
    plot_pca_data(X_train_pca, y_train, 'pca_all_features.pdf')
    train_acc, val_acc = evaluate_knn(X_train_pca, y_train, X_val_pca, y_val, 1)
    print(f"knn (k=1) Validation-Acc: {val_acc}")

    X_train_hist = get_histogram_data(X_train, 32)

    X_val_hist = get_histogram_data(X_val, 32)
    print('Apply PCA on color histogram data...')
    X_train_pca_hist, X_val_pca_hist = apply_pca(X_train_hist, X_val_hist, 2)
    plot_pca_data(X_train_pca_hist, y_train, 'pca_color_histogram.pdf')
    train_acc, val_acc = evaluate_knn(X_train_pca_hist, y_train, X_val_pca_hist, y_val, 1)
    print(f"knn (k=1) Validation-Acc: {val_acc}")

    train_acc, val_acc = train_custom_classifier(X_train, X_train_pca, X_train_pca_hist,
                                                 X_val, X_val_pca, X_val_pca_hist,
                                                 y_train, y_val)

    print(f"Validation Acc. for custom classifier: {val_acc}")


if __name__ == '__main__':
    main()
