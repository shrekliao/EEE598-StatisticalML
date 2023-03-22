import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

def load_mnist_dataset():
    mndata = MNIST("./data/mnist_data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, labels_train, X_test, labels_test

#def one_hot_encoder(labels_train):
#    return np.eye(len(set(labels_train)))[labels_train] #set remove replicate #

def accuracy_err(y_true, y_pred):
    return 1-np.mean(y_true == y_pred) #same rate

def train(X, Y, lamb):
    w_h = np.linalg.solve(X.T.dot(X) + lamb*np.eye(X.shape[1]), X.T.dot(Y))
    return w_h

def predict(W, X_prime):
    predictions = np.argmax(W.T.dot(X_prime.T), axis = 0)
    return predictions

def non_linear_transform(X, p, G=None, b=None):
    d = X.shape[1]
    if G is None:
        G = np.random.normal(0, 0.01, (p,d))
    if b is None:
        b = np.random.uniform(0, 2*np.pi, (p,))
    return np.cos(X.dot(G.T) + b), G, b

X_train, labels_train, X_test, labels_test = load_mnist_dataset()
Y_train = np.eye(len(set(labels_train)))[labels_train]  #one hot encode
#create random permutation
idx = np.random.permutation(X_train.shape[0])
train_split = int(X_train.shape[0]*0.8)
TRAIN = idx[:train_split]
VAL = idx[train_split:]
#create training set
X_train80 = X_train[TRAIN, :]
Y_train80 = Y_train[TRAIN] 
labels_train80 = labels_train[TRAIN] #train labels
#create validation set
X_val = X_train[VAL, :]
Y_val = Y_train[VAL]
labels_val = labels_train[VAL]
p_grid = np.arange(100, 1800, 200)
val_score = []
train_score = []

for p in p_grid:
    #Fit the model
    transformed_X_train80, G, b = non_linear_transform(X_train80, p)
    w_h_p = train(transformed_X_train80, Y_train80, 1e-4)
    #Predict for train set:
    labels_train_pred = predict(w_h_p, transformed_X_train80)
    train_score.append(accuracy_err(labels_train_pred, labels_train80))
    #Predict for validation set:
    transformed_X_val, _, _ = non_linear_transform(X_val, p, G, b) #use the same G,b
    labels_val_pred = predict(w_h_p, transformed_X_val)
    val_score.append(accuracy_err(labels_val_pred, labels_val))

plt.figure(figsize = (15, 8))
plt.plot(p_grid, train_score, '-o', label = 'train_score')
plt.plot(p_grid, val_score, '-o', label = 'val_score')
plt.title('Problem 5c')
plt.xlabel('p')
plt.ylabel('error')
plt.legend()
plt.show()
