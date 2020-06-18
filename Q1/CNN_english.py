#%tensorflow_version 1.x
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
from scipy.misc import imread,imresize
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import resize
import string
import cv2
from joblib import dump,load

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

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

print("test")
print(glob.glob('./drive/My Drive/AI_COM2028/training_type/a/*.png'))
def cleanName(n,char):
  n = n.replace('/drive/My Drive/AI_COM2028/training_type/'+char+'/','')
  n = n.replace('.png','')
  n = n.replace('.','')
  return n

allCharacters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9']
X_dictionarY_train = {}
X_train_list = []
Y_train = []

X_dictionary_test = {}
X_test_list = []
Y_test = []
for char in allCharacters:
 # print(char)
  try:
    print(X_dictionarY_train[char])
  except KeyError:
    X_dictionarY_train[char] = {}
    X_dictionary_test[char] = {}

  # get path
  alldir = glob.glob('./training_type/'+char+'/*.png')
  c = 1
  for i in alldir:
    # Test Data
    if(c%12 == 0):
      try:
        image = ski.imread(i,as_gray=True)
        #print(image.shape)
        i_new = cleanName(i,char)
        X_dictionary_test[char][i_new] = asarray(image)
        X_test_list.append(asarray(image))
        Y_test.append(char)
      except ValueError:
        print(i,"File corrupt error")
    # Traning Data
    else:
      try:
        image = ski.imread(i,as_gray=True)
        #print(image.shape)
        i_new = cleanName(i,char)
        X_dictionarY_train[char][i_new] = asarray(image)
        X_train_list.append(asarray(image))
        Y_train.append(char)
      except ValueError:
        print(i,"File corrupt error")
      
    c = c + 1


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
# Convert all labels

Y_test = [mappings(x) for x in Y_test]
Y_train = [mappings(x) for x in Y_train]

Y_test = asarray(Y_test)
Y_train = asarray(Y_train)

Y_train = Y_train.astype(np.int32)
Y_test = Y_test.astype(np.int32)
X_train = asarray(X_train_list)
X_test = asarray(X_test_list)

#print(Y_test)

# Fix the data
print(X_train.shape)
X_train = np.expand_dims(X_train,-1).astype(np.float32) / 255.0
X_test = np.expand_dims(X_test,-1).astype(np.float32) / 255.0





reset_graph()

X = tf.placeholder(tf.float32, shape=(None, 20,20,1), name = "X")
Y = tf.placeholder(tf.int32, shape=(None), name = "Y")
dropout_pb = tf.placeholder(tf.float32,name = "dropout_pb")

settings = {'conv1_kernal':(3,3),
           'conv1_stride':(1,1),
            'conv1_dim':64,
            'conv1_maxpool_size':(2,2),

            'conv1_maxpool_strides':(2,2),
            'conv2_kernal':(3,3),
           'conv2_stride':(1,1),
            'conv2_dim':128,
            'conv2_maxpool_size':(2,2),
            'conv2_maxpool_strides':(2,2),

            'conv3_kernal':(3,3),
           'conv3_stride':(2,2),
            'conv3_dim':64,
            'conv3_maxpool_size':(2,2),
            'conv3_maxpool_strides':(1,1),
            'fc':256,
            'output':len(allCharacters)
           }

n_conv1 = 64
n_hidden1 = 512
n_outputs = len(allCharacters)

#print("Total Characters" , len(allCharacters))
def CNN(X,dropout):
  conv1 = tf.layers.conv2d(X,settings['conv1_dim'], kernel_size = settings['conv1_kernal'], strides=settings['conv1_stride'], name="conv1",  activation=tf.nn.relu, padding="SAME")
  pool1 = tf.layers.max_pooling2d(conv1, pool_size=settings['conv1_maxpool_size'], strides=settings['conv1_maxpool_strides'])
  
  bn = tf.contrib.layers.batch_norm(pool1, is_training=True, scale=True, decay=0.99)

  conv2 = tf.layers.conv2d(bn,settings['conv2_dim'], kernel_size = settings['conv2_kernal'], strides=settings['conv2_stride'], name="conv2",  activation=tf.nn.relu,padding="SAME")
  pool2 = tf.layers.max_pooling2d(conv2, pool_size=settings['conv2_maxpool_size'], strides=settings['conv2_maxpool_strides'])
  
  conv3 = tf.layers.conv2d(pool2,settings['conv3_dim'], kernel_size = settings['conv3_kernal'], strides=settings['conv3_stride'], name="conv3",  activation=tf.nn.relu,padding="SAME")
  pool3 = tf.layers.max_pooling2d(conv3, pool_size=settings['conv3_maxpool_size'], strides=settings['conv3_maxpool_strides'])
  
  bn = tf.contrib.layers.batch_norm(pool3, is_training=True, scale=True, decay=0.99)
  bn = tf.layers.dropout(bn,dropout)

  flatten = tf.layers.flatten(bn)

  fc1 = tf.layers.dense(flatten, settings['fc'], name="fc1",activation=tf.nn.relu)
  fc1 = tf.layers.dropout(fc1,dropout)
  return tf.layers.dense(fc1,  settings['output'],name="outputs")




def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

EPOCHS = 20
batch_size = 50
learning_rate = 0.001
dropoutrate = 0.5
logits = CNN(X,dropoutrate)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=Y), name="loss")
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
            sess.run(training_op,feed_dict = {X:X_batch,Y:Y_batch})
          acc_batch = accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})
          acc_valid = accuracy.eval(feed_dict={X: X_test, Y: Y_test})
          print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)
      save_path = saver.save(sess, "./CNN_SaveEnglish.ckpt")
  
def applyModelOnSingleImage(image):
  with tf.Session() as sess:
    saver.restore(sess, "./CNN_Save.ckpt")
    img = ski.imread(image,as_gray=True)
    img = asarray(img)
    img = np.expand_dims(img,-1).astype(np.float32) / 255.0
    t = []
    t.append(img)
    Z = logits.eval(feed_dict={X: t})
    y = np.argmax(Z,axis =1)
    print(y)
    return (undoMapping(y[0]))

def applyModel(letters):
  with tf.Session() as sess:
    saver.restore(sess, "./CNN_SaveEnglish.ckpt")
    Z = logits.eval(feed_dict={X: letters})
    y = np.argmax(Z,axis =1)
    return y

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

def readOCR(filename, truth):
  extract = Extract_Letters()
  letters = asarray(extract.extractFile(filename))
  pyplot.imshow(letters[0])
  pyplot.show()
  letters = np.expand_dims(letters,-1)


  with tf.Session() as sess:
    saver.restore(sess, "./CNN_SaveEnglish.ckpt")
    Z = logits.eval(feed_dict={X: letters})
    output = np.argmax(Z,axis =1)
    with io.open(truth,'r') as f:
      contents = f.read().replace(" ", "").replace("\n","").strip().lower().replace("\"","").replace("’","").replace("’","").replace("“","").replace("”","").replace(",","").replace("\"","").replace(".","").replace("!","")
      print(contents)
      #print(output)
      output = [undoMapping(x) for x in output]
      truthCompare(output,contents)
      return output
runTraining()     
# # for i in glob.glob("./training_type/Ч/*.png"):

# for i in Y_train:
#   print(i)

#pred = readOCR("./ocr/testing/adobe.png","./ocr/testing/adobe_ground_truth.txt")
#print(toString(pred))

pred = readOCR("./ocr/testing/adobe.png","./ocr/testing/adobe_ground_truth.txt")
print(toString(pred))
# #runTraining()
