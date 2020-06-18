import sys, argparse, csv
import matplotlib
from matplotlib import pyplot
from numpy import genfromtxt
import np

with open('Handwritten.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    c = 0
    for row in reader:
        t = np.asarray(row)
        print("boo")
        if(c==1):
            break
        c = c + 1