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

def cleanTruth(art):
    art = art.replace(" ", "").replace("\n","").strip().lower().replace("\"","").replace("’","").replace("’","").replace("“","").replace("”","").replace(",","").replace("\"","").replace(".","").replace("!","")
    art = art.replace("(","").replace(")","").replace("-"," ").replace("'","")
    return art
def toString(d):
    s = ""
    for x in d:
        s += (x)
    return s
class Extract_Letters:
    def extractFile(self, filename):
        image = imread(filename, 1)
        # apply threshold in order to make the image binary
        bw = (image < 120).astype(np.float)

        # remove artifacts connected to image border
        cleared = bw.copy()
        # clear_border(cleared)

        # label image regions
        label_image = label(cleared, neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1

        letters = list()
        order = list()

        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            # skip small images
            if maxr - minr > len(image) / 250:  # better to use height rather than area.
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                order.append(region.bbox)

        # sort the detected characters left->right, top->bottom
        lines = list()
        first_in_line = ''
        counter = 0

        # worst case scenario there can be 1 character per line
        for x in range(len(order)):
            lines.append([])

        for character in order:
            if first_in_line == '':
                first_in_line = character
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
                first_in_line = character
                counter += 1
                lines[counter].append(character)

        for x in range(len(lines)):
            lines[x].sort(key=lambda tup: tup[1])

        final = list()
        prev_tr = 0
        prev_line_br = 0

        for i in range(len(lines)):
            for j in range(len(lines[i])):
                tl_2 = lines[i][j][1]
                bl_2 = lines[i][j][0]
                if tl_2 > prev_tr and bl_2 > prev_line_br:
                    tl, tr, bl, br = lines[i][j]
                    letter_raw = bw[tl:bl, tr:br]
                    letter_norm = resize(letter_raw, (20, 20))
                    final.append(letter_norm)
                    prev_tr = lines[i][j][3]
                if j == (len(lines[i]) - 1):
                    prev_line_br = lines[i][j][2]
            prev_tr = 0
            tl_2 = 0
           # print ('Characters recognized: ' + str(len(final)))
        return final
    
    def __init__(self):
        print("Extracting characters...")



print(glob.glob('./training_type/a/*.png'))


def cleanName(n, char):
    n = n.replace('./training_type/'+char+'/', '')
    n = n.replace('.png', '')
    n = n.replace('.', '')
    return n

allCharacters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9']
# To make everything easier I want to map every character to a number
def mappings(character):
    c = 0
    for i in allCharacters:
        if(character == i):
            break
        c = c + 1
    return c

def undoMapping(num):
  x = 0
  for i in allCharacters:
    if( x == num ):
      return allCharacters[x]
    x = x + 1
print("Total Character ", len(allCharacters))

totalClusters = len(allCharacters)
def trainSVM():
    loader_train = {}
    labels_train = []

    loader_test = {}
    labels_test = []

    Y_validate = []
    Y_train = []
    X_train = []
    X_validate = []

    for char in allCharacters:
        # print(char)
        # try:
        #   #print(loader_train[char])
        # except KeyError:
        #   loader_train[char] = {}
        #   loader_test[char] = {}

        # get path
        alldir = glob.glob('./training_type/'+char+'/*.png')
        c = 1
        for i in alldir:
            # Test Data
            if(c % 7 == 0):
                try:
                    # print(i)
                    image = ski.imread(i, as_gray=True)
                    image = resize(image, (200,200))

                    # print(image.shape)
                    i_new = cleanName(i, char)
                    #image_loaded = asarray(image).flatten()
                    image_hog = hog(image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
                    #loader_test[char][i_new] = asarray(image)
                    X_validate.append(image_hog)
                    Y_validate.append(char)
                except ValueError:
                    print(i,"File corrupt error")
            # # Traning Data
            else:
                try:
                    image = ski.imread(i, as_gray=True)
                    image = resize(image, (200,200))
                    # print(image.shape)
                    i_new = cleanName(i, char)
                    #loader_train[char][i_new] = asarray(image)
                    #image_loaded = asarray(image).flatten()
                    image_hog = hog(image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
                    X_train.append(image_hog)
                    Y_train.append(char)
                except ValueError:
                    print(i,"File corrupt error")
            c = c + 1
        print(char + " has been Loaded")


    print("All Images have been mounted")

    # X_train = asarray(X_train) / 255.0
    # X_validate = asarray(X_validate) / 255.0

    Y_validate = [mappings(x) for x in Y_validate]
    Y_train = [mappings(x) for x in Y_train]

    Y_validate = asarray(Y_validate)
    Y_train = asarray(Y_train)

    Y_train = Y_train.astype(np.int32)
    Y_validate = Y_validate.astype(np.int32)
    print(asarray(X_train).shape)
    print("Training Data: ", asarray(X_train).shape, "Validation Data: ", asarray(X_validate).shape)


    print('Training the SVM')
    clf = LinearSVC(dual=False,verbose=1)
    clf.fit(X_train, Y_train)
    y_knn_pred = clf.predict(X_validate)
    try:
        dump(clf,'svm_eng.joblib')
    except FileNotFoundError:
        io.open('svm_eng.joblib')
        dump(clf,'svm_eng.joblib')
    print(y_knn_pred)
    accuracy = accuracy_score(Y_validate, y_knn_pred)
    print("Accuracy (Validation): ", (accuracy * 100))
    print("Training Data: ", asarray(X_train).shape, "Validation Data: ", asarray(X_validate).shape)

    return clf

def truthCompare(output,truth):
  truth_array = list(truth)
  acc = 0
  if(len(truth_array) == len(output)):
    acc = accuracy_score(truth_array,output)
  elif(len(truth_array) > len(output)):
    truth_array = truth_array[:len(output)]
    acc = accuracy_score(truth_array,output)
  elif(len(truth_array) < len(output)):
    output = output[:len(truth_array)]
    acc = accuracy_score(truth_array,output)
  else:
    acc = accuracy_score(truth_array,output)
  print("Accuracy of OCR: ",(acc * 100),"%")

def readOCR(filename, truth, clf):
  extract = Extract_Letters()
  hog_converted = list()
  letters = (extract.extractFile(filename))
  for i in letters:
    image = resize(i, (200,200))
    image_hog = hog(image, orientations=12, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
    hog_converted.append(image_hog)
  y_knn_pred = clf.predict(hog_converted)
  y_knn_pred = [undoMapping(x) for x in y_knn_pred]
  with io.open(truth,'r') as f:
    contents = f.read().replace(" ", "").replace("\n","").strip().lower().replace("\"","").replace("’","").replace("’","").replace("“","").replace("”","").replace(",","").replace("\"","").replace(".","").replace("!","")
    #print(contents)
    truthCompare(y_knn_pred,contents)
  return y_knn_pred
  #print(letters)


def loadModel():
    return load("svm_eng.joblib")



clf = trainSVM()
#clf = loadModel()
print("Adobe")
pred = readOCR("./ocr/testing/adobe.png","./ocr/testing/adobe_ground_truth.txt", clf)
#print(toString(pred))
print("Shazam")
pred = readOCR("./ocr/testing/shazam.png","./ocr/testing/shazam_ground_truth.txt", clf)
#print(toString(pred))


# pyplot.imshow(X_train[0])
# pyplot.show()
