# Cleaning NIST dataset.

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
import PIL
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageDraw, ImageOps, ImageMath
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


# http://hanyu.iciba.com/zt/3500.html
data = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9']
t = []
h = []
path = "./nist_28"
l = 0
limit = 30
for i in data:
    allImages = glob.glob("./nist/"+i+"/*.png")
    counter = 0 
    fullpath = path + "/" + i
    for dir_ in allImages:
        x = Image.open(dir_).resize((28,28))
        newImage = x
        #newImage = PIL.ImageOps.invert(x)
        try:
            newImage.save(fullpath + "/" + str(l)+".png",quality=95)
        except FileNotFoundError:
            print("Creating File")
            if not (os.path.exists(path)):
                os.mkdir(path)
            if not (os.path.exists(fullpath)):
                os.mkdir(fullpath)
            io.open(fullpath + '/'+str(l)+".png",'w')
            newImage.save(fullpath +"/" + str(l)+".png",quality=95)
        l = l + 1
        counter = counter + 1
        #if(counter % 40 == 0):
            #break
    print(i + " Completed")
    
print(len(data))
