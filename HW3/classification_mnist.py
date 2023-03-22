import numpy as np
import random
import matplotlib.pyplot as plt
from mnist.loader import MNIST

mndata = MNIST("./data/mnist_data/")
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train / 255.0
X_test = X_test / 255.0


#convert training labels to one hot
Y_train = np.zeros((X_train.shape[0], 10))
for i, digit in enumerate(labels_train):
        Y_train[i, digit] = 1

def grad_J(X, y, W, lam=0):
    yhat = np.dot(X, W)
    return -1 * np.dot(X.T, y - yhat) + 2 * lam * W


def J_function(X, y, W):
    return 0.5 * np.sum(np.square(y - np.dot(X, W)))

def grad_L(X,y, W):
    yhat = np.exp(np.dot(X, W))
    summand = np.expand_dims(np.sum(yhat, axis=1), axis=1)
    yhat = yhat / summand

    return -1 * np.dot(X.T, y - yhat)

def L_function(X,y,W):
    n = X.shape[0]
    Wy = np.dot(W, y.T)
    Wyx = np.asarray([np.dot(X[i, :], Wy[:, i]) for i in range(n)])
    summand = np.sum(np.exp(np.dot(W.T, X.T)) , axis=0)
    return -1 * np.sum(Wyx - np.log(summand))


def error_rate(X, labels, W):
    predictions = np.argmax(np.dot(W.T, X.T), axis=0)
    return np.sum(np.where(predictions != labels, 1, 0)) / len(labels)


def gradient_descent(x_init, gradient_function, eta=0.1, delta=1e-3, X=None, y=None):
    x = x_init
    grad = gradient_function(x)
    it = 0
    while (np.amax(np.abs(grad)) > delta and (it < 1500 )): #STOP
        it += 1
        print("it: ",it)
        x = x - eta * grad
        print("amax(abs(grad): ",np.amax(np.abs(grad)))
        print("J_function: ",J_function(X, y, x))
        print("L_function: ",L_function(X, y, x))
        grad = gradient_function(x)
    return x

# Problem 2c)
n = X_train.shape[0]
d = X_train.shape[1] # 28 x 28 = 784
k = 10 

# define initial
W_init_L = np.zeros((d, k))
W_init_J = np.zeros((d, k))
gradient_function_J = lambda W: grad_J(X=X_train, y=Y_train, W=W, lam=lam)
gradient_function_L = lambda W: grad_L(X=X_train, y=Y_train, W=W)


# J function
delta = 1e-3 * n #60
print("delta: ", delta)
eta = 7e-7 # learning rate
lam = 0

J_best_train = gradient_descent(W_init_J, gradient_function_J, eta, delta, X=X_train, y=Y_train)

J_training_error_rate = error_rate(X_train, labels_train, J_best_train)
J_testing_error_rate = error_rate(X_test, labels_test, J_best_train)

print("J train:", J_training_error_rate)
print("J test: ", J_testing_error_rate)


# L function
delta = 1e-3 * n #60
eta = 7e-7 # learning rate
lam = 0
L_best_train = gradient_descent(W_init_L, gradient_function_L, eta, delta, X=X_train, y=Y_train)

L_training_error_rate = error_rate(X_train, labels_train, L_best_train)
L_testing_error_rate = error_rate(X_test, labels_test, L_best_train)

print("L train:", L_training_error_rate)
print("L test: ", L_testing_error_rate)
