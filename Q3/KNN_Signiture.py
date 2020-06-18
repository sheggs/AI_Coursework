# trainKNN() to train model
# loadKNN() to load the trained model


from PIL import Image, ImageDraw
from math import sqrt
from numpy import asarray
import random
from urllib.request import urlopen
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as ski
# %matplotlib inline
from sklearn.cluster import KMeans
import glob
import os
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageDraw
from matplotlib import image
from numpy import asarray
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches
import io
from scipy.misc import imread, imresize
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import resize
import string
import cv2
from joblib import dump, load


def fixPadding(vector, width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:width[0]] = pad_value
    vector[-width[1]:] = pad_value


def apply(data):
    temp = []
    for i in range(len(data)):
        temp.append(np.pad(data[i], 4, fixPadding, padder=0))
    return temp


def cleanTruth(art):
    art = art.replace(" ", "").replace("\n", "").strip().lower().replace("\"", "").replace("’", "").replace(
        "’", "").replace("“", "").replace("”", "").replace(",", "").replace("\"", "").replace(".", "").replace("!", "")
    art = art.replace("(", "").replace(")", "").replace(
        "-", " ").replace("'", "")
    return art


def toString(d):
    s = ""
    for x in d:
        s += x
    return s


class Extract_Letters:
    def extractFile(self, filename):
        image = imread(filename, 1)
        # apply threshold in order to make the image binary
        image = resize(image, (50, 50), anti_aliasing=True)
        bw = (image < 120).astype(np.float)
        output = []
        output.append(asarray(bw).flatten())
        return output

    def __init__(self):
        print("Getting Signiture")


signitures = ['Mary_Jane', 'John_Doe']


def trainKNN():
    loader_train = {}
    labels_train = []

    loader_test = {}
    labels_test = []

    Y_validate = []
    Y_train = []
    X_train = []
    X_validate = []
    X_train_list = []
    X_test_list = []
    Y_test = []


    for char in signitures:
        # get path
        alldir = glob.glob('./training_type/'+char+'/*')
        c = 1
        counter = 0
        for i in alldir:
            # To prevent overfitting we limit to 100 images per class. [Insted implemented in the data cleansing python file]
            # Test Data
            if(c % 4 == 0):
                try:
                    image = ski.imread(i, as_gray=True)
                    image = resize(image, (50, 50), anti_aliasing=True)
                    # pyplot.imshow(image)
                    # pyplot.show()
                    X_test_list.append(asarray(image).flatten())
                    Y_test.append(char)
                except ValueError:
                    print(i, "File corrupt error")
            # Traning Data
            else:
                try:
                    image = ski.imread(i, as_gray=True)
                    image = resize(image, (50, 50), anti_aliasing=True)
                    # pyplot.imshow(image)
                    # pyplot.show()
                    X_train_list.append(asarray(image).flatten())
                    Y_train.append(char)
                except ValueError:
                    print(i, "File corrupt error")

            c = c + 1
        print("Loaded " + char)

    print("All Images have been mounted")

    Y_test = asarray(Y_test)
    Y_train = asarray(Y_train)

    X_train = asarray(X_train_list)
    X_test = asarray(X_test_list)

    # print(Y_test)

    # Fix the data
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    print(X_train.shape)
    print("Training Data: ", X_train.shape,
          "Validation Data: ", X_test.shape)

    # Passing it into KNN
    knn_clf = KNeighborsClassifier(
        n_jobs=-1, weights='distance', n_neighbors=len(signitures))
    knn_clf.fit(X_train, Y_train)
    y_knn_pred = knn_clf.predict(X_test)
    try:
        dump(knn_clf, 'knn_signiture.joblib')
    except FileNotFoundError:
        io.open('knn_signiture.joblib')
        dump(knn_clf, 'knn_signiture.joblib')
    print(y_knn_pred)
    accuracy = accuracy_score(Y_test, y_knn_pred)
    print("Accuracy (Validation): ", (accuracy * 100))
    return knn_clf


def truthCompare(output, truth):
    truth_array = list(truth)
    acc = 0
    if(len(truth_array) > len(output)):
        truth_array = truth_array[:len(output)]
        acc = accuracy_score(truth_array, output)
    elif(len(truth_array) < len(output)):
        output = output[:len(truth_array)]
        acc = accuracy_score(truth_array, output)
    else:
        acc = accuracy_score(truth_array, output)

    print("Accuracy of OCR: ", (acc * 100), "%")


def readOCR(filename, truth, knn_clf):
    extract = Extract_Letters()
    letters = asarray(extract.extractFile(filename))
    y_knn_pred = knn_clf.predict(letters)
    with io.open(truth, 'r') as f:
        contents = f.read()
        if(y_knn_pred[0] == contents.strip()):
            print(filename + " Matches Predicted: " + y_knn_pred[0] + " Actual: "+ contents)
            return True
        else: 
            print(filename+ " Failed Predicted: " + y_knn_pred[0] + " Actual: "+ contents)
            return False
    return y_knn_pred
    # print(letters)


def loadModel():
    return load("knn_signiture.joblib")


knn_clf = trainKNN()
knn_clf = loadModel()


pred1 = readOCR("./ocr/ocr1.png","./ocr/ocr_1.txt",knn_clf)
pred2 = readOCR("./ocr/ocr2.png","./ocr/ocr_2.txt",knn_clf)
pred3 = readOCR("./ocr/ocr3.png","./ocr/ocr_3.txt",knn_clf)
pred4 = readOCR("./ocr/ocr4.png","./ocr/ocr_4.txt",knn_clf)
pred5 = readOCR("./ocr/ocr5.png","./ocr/ocr_5.txt",knn_clf)
pred6 = readOCR("./ocr/ocr6.png","./ocr/ocr_6.txt",knn_clf)
# pyplot.imshow(X_train[0])
# pyplot.show()
