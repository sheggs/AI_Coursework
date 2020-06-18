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
from joblib import dump,load

def cleanTruth(art):
    art = art.replace(" ", "").replace("\n","").strip().lower().replace("\"","").replace("’","").replace("’","").replace("“","").replace("”","").replace(",","").replace("\"","").replace(".","").replace("!","")
    art = art.replace("(","").replace(")","").replace("-"," ").replace("'","")
    return art
def toString(d):
    s = ""
    for x in d:
        s += x
    return s
class Extract_Letters:
    def extractFile(self, filename):
        image = imread(filename, 1)
        print(image)
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







print("test")
print(glob.glob('./training_type/a/*.png'))


def cleanName(n, char):
    n = n.replace('./training_type/'+char+'/', '')
    n = n.replace('.png', '')
    n = n.replace('.', '')
    return n

allCharacters = ['a','b','c','ç','d','e','f','g','ğ','h','ı','i','j','k','l','m','n','o','ö','p','r','s','ş','t','u','ü','v','y','z','1','2','3','4','5','6','7','8','9']

print("Total Character ", len(allCharacters))

totalClusters = len(allCharacters)
def trainKNN():
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
        alldir = glob.glob('./training_type/turkish_'+char+'/*.png')
        c = 1
        for i in alldir:
            # Test Data
            if(c % 7 == 0):
                try:
                    # print(i)
                    image = ski.imread(i, as_gray=True)
                    # print(image.shape)
                    i_new = cleanName(i, char)
                    #loader_test[char][i_new] = asarray(image)
                    X_validate.append(asarray(image).flatten())
                    Y_validate.append(char)
                except ValueError:
                    print(i,"File corrupt error")
            # # Traning Data
            else:
                try:
                    image = ski.imread(i, as_gray=True)
                    # print(image.shape)
                    i_new = cleanName(i, char)
                    #loader_train[char][i_new] = asarray(image)
                    X_train.append(asarray(image).flatten())
                    Y_train.append(char)
                except ValueError:
                    print(i,"File corrupt error")
            c = c + 1
        print(char + " has been Loaded")


    print("All Images have been mounted")

    X_train = asarray(X_train) / 255.0
    X_validate = asarray(X_validate) / 255.0

    print(X_train.shape)
    print(X_train)
    print("Training Data: ", X_train.shape, "Validation Data: ", X_validate.shape)

    # Passing it into KNN
    knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=totalClusters)
    knn_clf.fit(X_train, Y_train)
    y_knn_pred = knn_clf.predict(X_validate)
    try:
        dump(knn_clf,'knn_turkish.joblib')
    except FileNotFoundError:
        io.open('knn_turkish.joblib')
        dump(knn_clf,'knn_turkish.joblib')
    print(y_knn_pred)
    accuracy = accuracy_score(Y_validate, y_knn_pred)
    print("Accuracy (Validation): ", (accuracy * 100))
    return knn_clf

def truthCompare(output,truth):
  truth_array = list(truth)
  acc = 0
  if(len(truth_array) > len(output)):
    truth_array = truth_array[:len(output)]
    acc = accuracy_score(truth_array,output)
  elif(len(truth_array) < len(output)):
    output = output[:len(truth_array)]
    acc = accuracy_score(truth_array,output)
  else:
    acc = accuracy_score(truth_array,output)

  print("Accuracy of OCR: ",(acc * 100),"%")

def readOCR(filename, truth, knn_clf):
  extract = Extract_Letters()
  letters = asarray(extract.extractFile(filename))
  test = letters[0] 
  #print(test)
  #pyplot.imshow(letters[0])
  #pyplot.show()
  letters = (letters.reshape(-1,400))
  y_knn_pred = knn_clf.predict(letters)
  with io.open(truth,'r') as f:
    contents = f.read().replace(" ", "").replace("\n","").strip().lower().replace("\"","").replace("’","").replace("’","").replace("“","").replace("”","").replace(",","").replace("\"","").replace(".","").replace("!","")
    truthCompare(y_knn_pred,contents)
  return y_knn_pred
  #print(letters)


def loadModel():
    return load("knn_turkish.joblib")





knn_clf = trainKNN()
pred = readOCR("./ocr/testing/turkish_testing1.png","./ocr/testing/turkish_testing1.txt", knn_clf)
print(toString(pred))



# pyplot.imshow(X_train[0])
# pyplot.show()
