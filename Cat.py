import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import scipy
# import scipy.misc
from PIL import Image
# from scipy import ndimage
import sys
import os

W,b,Z,A = [], [], [], []
images = []
m = 3000
dev_size = 50
alpha = 0.5
num_px = 100
layer_dim = [num_px * num_px * 3,40,40,40,1]
X,Y,A = [],[],[]
dev_X, dev_Y = [], []

def input():

    global X
    global Y
    global dev_X
    global dev_Y

    n = 0
    count = 0
    for f in os.listdir("train"):
        if count >= m + dev_size:
            break
        count += 1
        if f == '.DS_Store':
            continue
        im = Image.open("train/" + f)
        im = im.resize((num_px, num_px))
        images.append(im)
        pix = np.array(im)
        pix = pix.reshape(num_px * num_px * 3, 1)
        X.append(pix)
        if f[0] == 'd':
            Y.append(1)
        else:
            Y.append(0)
    count = 0
    for i in Y:
        count += i

    dev_X, dev_Y = format_data(X[m:], Y[m:], dev_size)
    X,Y = format_data(X[:m], Y[:m], m)

def format_data(X,Y,size):
    X = np.array(X)
    Y = np.array(Y)
    Y = Y.reshape(size,1)
    X = np.squeeze(X)
    X = X.T
    X = X/255
    Y = Y.T
    return (X,Y)


def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid(Z):
    return 1 / (1+np.exp(-Z))

def derivative(function, layer):
    if function == 'tanh':
        if layer == -1:
            return 1- np.power(X, 2)
        return 1 - np.power(A[layer],2)
    # if function == 'sigmoid':


def initialize():
    global W
    global b
    W = []
    b = []
    for i in range(1,len(layer_dim)):
        W.append(np.random.randn(layer_dim[i],layer_dim[i-1]))
        b.append(np.zeros((layer_dim[i],1)))

def forward_feed(X):
    global Z
    global A
    Z,A = [],[]
    #print(W == None, X == None, b == None)
    Z.append(np.dot(W[0],X) + b[0])
    A.append(tanh(Z[0]))
    for i in range(1,len(layer_dim)-1):
        Z.append(np.dot(W[i],A[i-1]) + b[i])
        if i == len(layer_dim)-2:
            A.append(sigmoid(Z[i]))
        else:
            A.append(tanh(Z[i]))

def backward_propagation():
    global W
    global b

    n = len(layer_dim)
    dW,db = [None]*(n-1),[None]*(n-1)
    dZ = [None] * (n - 1)
    dZ_prev = A[n-2] - Y

    dW[-1] = (1.0/m) * np.dot(dZ_prev,A[-1].T)
    db[-1] = np.sum(dZ_prev,axis = 1)


    for i in reversed(range(n-2)):

        dZ[i] = np.dot(W[i+1].T,dZ_prev) * derivative("tanh",i)
        if i == 0:
            dW[i] = (1.0/m) * np.dot(dZ[i],X.T)
        else:
            dW[i] = (1.0/m) * np.dot(dZ[i], A[i-1].T)

        db[i] = (1/m) * np.sum(dZ[i],axis = 1, keepdims = True)
        dZ_prev = dZ[i]
        W[i] -= dW[i] * alpha
        b[i] -= db[i] * alpha

def cost():
    return (-1.0/m) * np.sum(Y * np.log(A[-1]) + (1-Y) * np.log(1-A[-1]))


def predict(img):
    forward_feed(img)
    if np.round(A[-1] > 0.5):
        print("dog")
        return 1;
    else:
        print("cat")
        return 0;

def predict_test(img_file):
    im = Image.open(img_file)
    im = im.resize((num_px, num_px))
    #im.show()
    test = np.array(im)
    test = test.reshape(num_px * num_px * 3, 1)

    print(test.shape)
    test = test/255
    predict(test)
    #if (A[-1])

def training_accuracy():
    forward_feed(X)
    correct = 0
    for i in range(m):
        if (np.round(A[-1][0,i], decimals = 0) == Y[0,i]):
            correct += 1
    return float(correct)/m

def dev_accuracy():
    forward_feed(dev_X)
    correct = 0
    for i in range(dev_size):
        if (np.round(A[-1][0,i], decimals = 0) == dev_Y[0,i]):
            correct += 1
    return float(correct)/dev_size

def store_weights():
    f = open('cat_weights.txt', 'w')
    #weights = np.concatenate(W, axis=0 )
    #np.save('cat_weights.txt', weights)
    f.write(str(len(layer_dim)) + "\n")
    for i in layer_dim:
        f.write(str(i) + "\n")
    for i in range(len(layer_dim) - 1):
        for j in range(len(W[i])):
            for k in range(len(W[i][j])):
                f.write(str(W[i][j,k]) + " ")
            f.write("\n")

    for i in range(len(layer_dim)-1):
        for j in b[i]:
            for k in j:
                f.write(str(k) + " ")
        f.write("\n")


def load_weights():
    global W
    global b
    f = open('cat_weights.txt', 'r')
    layer_dim = []
    n = int(f.readline())
    W = []
    b = []

    for i in range(n):
        layer_dim.append(int(f.readline()))

    for i in range(1, n):
        W.append(np.array([]))

        for j in range(layer_dim[i]):
            line = [float(k) for k in f.readline().split()]
            W[-1] = np.append(W[-1], np.array(line))

        W[-1] = W[-1].reshape(layer_dim[i], layer_dim[i-1])

    for i in range(1, n):
        b.append(np.array([float(k) for k in f.readline().split()]))
        b[-1] = b[-1].reshape(layer_dim[i], 1)

input()
load_weights()

# initialize()

for i in range(500):
    forward_feed(X)
    backward_propagation()
    if i % 2 == 0:
        print(cost())

store_weights()

print()
print('training accuracy = ' + training_accuracy())
print('dev accuracy = ' + dev_accuracy())
