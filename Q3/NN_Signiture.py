# %tensorflow_version 1.x
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import asarray
import os
import io
from PIL import Image, ImageDraw
from math import sqrt
import random
from urllib.request import urlopen
import numpy as np
import glob
# Common imports
import numpy as np
import os
import skimage.io as ski
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
from skimage.transform import resize, rescale
import string
import cv2
from joblib import dump, load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print("Opened")
def toString(d):
    s = ""
    for x in d:
        s += x
    return s


class Extract_Letters:
    def extractFile(self, filename):
        image = imread(filename, 1)
        image = resize(image,(50,50), anti_aliasing=True)        # apply threshold in order to make the image binary
        bw = (image < 120).astype(np.float)
        return bw

    def __init__(self):
        print("Getting Signiture")

# to make this notebook's output stable across runs


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)




signitures = ['Mary_Jane','John_Doe']


X_dictionarY_train = {}
X_train_list = []
Y_train = []

X_dictionary_test = {}
X_test_list = []
Y_test = []
for char in signitures:
 # print(char)
    try:
        print(X_dictionarY_train[char])
    except KeyError:
        X_dictionarY_train[char] = {}
        X_dictionary_test[char] = {}

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
                image = resize(image,(50,50), anti_aliasing=True)
                #pyplot.imshow(image)
                #pyplot.show()
                X_test_list.append(asarray(image))
                Y_test.append(char)
            except ValueError:
                print(i, "File corrupt error")
        # Traning Data
        else:
            try:
                image = ski.imread(i, as_gray=True)
                image = resize(image,(50,50), anti_aliasing=True)
                #pyplot.imshow(image)
                #pyplot.show()
                X_train_list.append(asarray(image))
                Y_train.append(char)
            except ValueError:
                print(i, "File corrupt error")

        c = c + 1
    print("Loaded " + char)


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
# Convert all labels



Y_test = [mappings(x) for x in Y_test]
Y_train = [mappings(x) for x in Y_train]

Y_test = asarray(Y_test)
Y_train = asarray(Y_train)

Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)
X_train = asarray(X_train_list)
X_test = asarray(X_test_list)



# print(Y_test)

# Fix the data
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
X_train = np.expand_dims(X_train, -1).astype(np.float32) / 255.0
X_test = np.expand_dims(X_test, -1).astype(np.float32) / 255.0
X_train = (X_train.reshape(-1,50*50))
X_test = (X_test.reshape(-1,50*50))


reset_graph()

X = tf.placeholder(tf.float32, shape=(None, 50*50), name="X")
Y = tf.placeholder(tf.int32, shape=(None), name="Y")
dropout_pb = tf.placeholder(tf.float32, name="dropout_pb")
settings = {'dense_1':512,
            'dense_2':200,
            'output': len(signitures)
            }


#print("Total Characters" , len(allCharacters))


def NeuralNetwork(X,dropout):
    hidden1 = tf.layers.dense(X, settings['dense_1'], name="hidden1", activation=tf.nn.relu)
    bn = tf.contrib.layers.batch_norm(hidden1, is_training=True, scale=True, decay=0.99)
    drop = tf.layers.dropout(bn,dropout)

    hidden2 = tf.layers.dense(drop,settings['dense_2'],name="hidden2", activation=tf.nn.relu)
    bn = tf.contrib.layers.batch_norm(hidden2, is_training=True, scale=True, decay=0.99)
    drop = tf.layers.dropout(bn,dropout)

    return tf.layers.dense(drop ,  settings['output'],name="outputs", activation=tf.nn.softmax)



def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def fixPadding(vector, width, iaxis, kwargs):
    pad_value = kwargs.get('padder',10)
    vector[:width[0]] = pad_value
    vector[-width[1]:] = pad_value

def apply(data):
    temp = []
    for i in range(len(data)):
        temp.append(np.pad(data[i],4,fixPadding,padder=0))
    return temp

EPOCHS = 15
batch_size = 2
learning_rate = 0.001
dropoutrate = 0.25
logits = NeuralNetwork(X, dropoutrate)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=Y), name="loss")
correct = tf.nn.in_top_k(logits, Y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
saver = tf.train.Saver()


def runTraining():
    print("Training....")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for epoch in range(EPOCHS):
            for X_batch, Y_batch in shuffle_batch(X_train, Y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_test, Y: Y_test})
            print(epoch, "Batch accuracy:", acc_batch,
                  "Validation accuracy:", acc_valid)
        save_path = saver.save(sess, "./CNN_SIGNITURE.ckpt")


def applyModelOnSingleImage(image):
    with tf.Session() as sess:
        saver.restore(sess, "./CNN_Save.ckpt")
        img = ski.imread(image, as_gray=True)
        img = asarray(img)
        img = np.expand_dims(img, -1).astype(np.float32) / 255.0
        t = []
        t.append(img)
        Z = logits.eval(feed_dict={X: t})
        y = np.argmax(Z, axis=1)
        print(y)
        return (undoMapping(y[0]))


def applyModel(letters):
    with tf.Session() as sess:
        saver.restore(sess, "./CNN_SIGNITURE.ckpt")
        Z = logits.eval(feed_dict={X: letters})
        y = np.argmax(Z, axis=1)
        return y


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


def readOCR(filename, truth):
    extract = Extract_Letters()
    letters = asarray(extract.extractFile(filename))
    letters = np.expand_dims(letters, -1).astype(np.float32) / 255.0
    letters = (letters.reshape(-1,50*50))
    with tf.Session() as sess:
        saver.restore(sess, "./CNN_SIGNITURE.ckpt")
        Z = logits.eval(feed_dict={X: letters})
        output = np.argmax(Z, axis=1)
        with io.open(truth, 'r') as f:
            contents = f.read()
            output = [undoMapping(x) for x in output]
            if(output[0].strip() == contents.strip()):
                print(filename + " Matches Predicted: " + output[0] + " Actual: "+ contents)
                return True
            else: 
                print(filename+ " Failed Predicted: " + output[0] + " Actual: "+ contents)
                return False


runTraining()
# # for i in glob.glob("./training_type/Ч/*.png"):

# for i in Y_train:
#   print(i)

#pred = readOCR("./ocr/testing/adobe.png","./ocr/testing/adobe_ground_truth.txt")
# print(toString(pred))


pred1 = readOCR("./ocr/ocr1.png","./ocr/ocr_1.txt")
pred2 = readOCR("./ocr/ocr2.png","./ocr/ocr_2.txt")
pred3 = readOCR("./ocr/ocr3.png","./ocr/ocr_3.txt")
pred4 = readOCR("./ocr/ocr4.png","./ocr/ocr_4.txt")
pred5 = readOCR("./ocr/ocr5.png","./ocr/ocr_5.txt")
pred6 = readOCR("./ocr/ocr6.png","./ocr/ocr_6.txt")

# print("NIST2")
# pred = readOCR("./ocr/testing/MY_HANDWRITING_TEST_1.png","./ocr/testing/NIST_2_ground_truth.txt")
# print(toString(pred))
# # # #runTraining()
