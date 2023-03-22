import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def load_mnist_dataset():
    mndata = MNIST("./data/mnist_data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test

def train (X, Y, lamb):
    n, d = X.shape
    a = np.dot (X.T, X) + lamb * np.eye(d) #one-hot
    b = np.dot (X.T, Y)
    w_h = np.linalg.solve(a, b)
    return w_h

def one_hot ( length , index ):
    arr = np.zeros(length)
    arr[index] = 1
    return arr

def predict (W, data , labelDim):
    predictions = np. dot (data , W)
    # pick out only the maximum values
    maxPredictions = np. argmax ( predictions , axis =1)
    classifications = np. array ([one_hot ( labelDim , y) for y in maxPredictions])
    return classifications

def calc_success_ratio (W, data , labels):
    n, d = data.shape
    labelDim = labels.shape[-1]
    wrong = np. sum (np. abs ( predict (W, data , labelDim ) - labels )) / 2.0
    # 2 is required because abs value will contribute double to the sum
    Wrongratio = wrong /n
    Correctratio = 1 - Wrongratio
    return Correctratio , Wrongratio

lamb = 1e-4
trainData, trainLabels, testData, testLabels = load_mnist_dataset ()
n, d = trainData . shape
labelDim = trainLabels.max() + 1
trainOneHot = np. array ([ one_hot ( labelDim , y) for y in trainLabels ])
testOneHot = np. array ([ one_hot ( labelDim , y) for y in testLabels ])
wHat = train ( trainData , trainOneHot , lamb )
trainErr = calc_success_ratio (wHat , trainData , trainOneHot )[ -1]
testErr = calc_success_ratio (wHat , testData , testOneHot )[ -1]
print (f" Train error : { trainErr }")
print (f" Test error : { testErr }")