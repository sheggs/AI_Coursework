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
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import resize
import string
import cv2
from joblib import dump,load


DATASET = "./dataset"
#NIST_DATASET = "./nist_git/20x20"

# apply threshold in order to make the image binaryNIST_DATASET
data = ['Mary_Jane','John_Doe']


outputPath = "./training_type"
label = 0
for i in data:
    allImages = glob.glob(DATASET+"/"+i+"/*")
    print(allImages)
    for j in allImages:
        image = imread(j, 1)
        bw = (image < 120).astype(np.float)
        fullPath = outputPath +"/" + i
        try:
            imsave(outputPath+"/"+ i +"/"+str(label)+".png",bw)
        except FileNotFoundError:
            if not (os.path.exists(outputPath)):
                os.mkdir(outputPath)
            if not (os.path.exists(fullPath)):
                os.mkdir(fullPath)
            io.open(fullPath + '/'+str(label)+".png",'w')
            imsave(outputPath+"/"+ i +"/"+str(label)+".png",bw)

        label = label + 1

    print(i + " is done")


print(bw)
pyplot.imshow(bw)
pyplot.show()