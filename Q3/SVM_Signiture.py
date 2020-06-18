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
from scipy.misc import imread,imresize
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import resize
import string
import cv2
from scipy import ndimage
from skimage.transform import resize
import imageio
from skimage import img_as_float, color, exposure
from skimage.feature import peak_local_max, hog
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
from joblib import dump,load

# To make everything easier I want to map every character to a number
def mappings(name):
    c = 0
    for i in signitures:
        if(name == i):
            break
        c = c + 1
    return c


def undoMapping(num):
    x = 0
    for i in signitures:
        if(x == num):
            return signitures[x]
        x = x + 1



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
        image = resize(image, (200, 200), anti_aliasing=True)
        # apply threshold in order to make the image binary
        bw = (image < 120).astype(np.float)
        output = []
        output.append(asarray(bw))
        return output

    def __init__(self):
        print("Getting Signiture")


signitures = ['Mary_Jane', 'John_Doe']


def trainSVM():
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
                    image = resize(image, (200, 200), anti_aliasing=True)
                    # pyplot.imshow(image)
                    # pyplot.show()
                    image_hog = hog(image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))

                    X_test_list.append(image_hog)
                    Y_test.append(char)
                except ValueError:
                    print(i, "File corrupt error")
            # Traning Data
            else:
                try:
                    image = ski.imread(i, as_gray=True)
                    image = resize(image, (200, 200), anti_aliasing=True)
                    image_hog = hog(image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))

                    # pyplot.imshow(image)
                    # pyplot.show()
                    X_train_list.append(image_hog)
                    Y_train.append(char)
                except ValueError:
                    print(i, "File corrupt error")

            c = c + 1
        print("Loaded " + char)

    print("All Images have been mounted")
    Y_test = [mappings(x) for x in Y_test]
    Y_train = [mappings(x) for x in Y_train]

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

    print('Training the SVM')
    clf = LinearSVC(dual=False,verbose=1)
    clf.fit(X_train, Y_train)
    y_svm_pred = clf.predict(X_test)
    try:
        dump(clf, 'svm_signiture.joblib')
    except FileNotFoundError:
        io.open('svm_signiture.joblib')
        dump(clf, 'svm_signiture.joblib')
    print(y_svm_pred)
    accuracy = accuracy_score(Y_test, y_svm_pred)
    print("Accuracy (Validation): ", (accuracy * 100))
    return clf


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


def readOCR(filename, truth, clf):
    extract = Extract_Letters()
    letters = asarray(extract.extractFile(filename))
    hog_converted = list()
    for i in letters:
        image_hog = hog(i, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
        hog_converted.append(image_hog)
    y_svm_pred = clf.predict(hog_converted)
    with io.open(truth, 'r') as f:
        contents = f.read()
        print(y_svm_pred)
        unmapped = undoMapping(y_svm_pred[0])
        if(unmapped == contents.strip()):
            print(filename + " Matches Predicted: " + unmapped + " Actual: "+ contents)
            return True
        else: 
            print(filename+ " Failed Predicted: " + unmapped + " Actual: "+ contents)
            return False
    return y_svm_pred
    # print(letters)


def loadModel():
    return load("svm_signiture.joblib")


clf = trainSVM()
clf = loadModel()


pred1 = readOCR("./ocr/ocr1.png","./ocr/ocr_1.txt",clf)
pred2 = readOCR("./ocr/ocr2.png","./ocr/ocr_2.txt",clf)
pred3 = readOCR("./ocr/ocr3.png","./ocr/ocr_3.txt",clf)
pred4 = readOCR("./ocr/ocr4.png","./ocr/ocr_4.txt",clf)
pred5 = readOCR("./ocr/ocr5.png","./ocr/ocr_5.txt",clf)
pred6 = readOCR("./ocr/ocr6.png","./ocr/ocr_6.txt",clf)
# pyplot.imshow(X_train[0])
# pyplot.show()
