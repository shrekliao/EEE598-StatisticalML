import numpy as np
import random
import matplotlib . pyplot as plt
#from mnist . loader import MNIST
from mnist import MNIST

mndata = MNIST ("./data/mnist_data/")
X_train , labels_train = map (np.array , mndata . load_training ())
X_test , labels_test = map (np.array , mndata . load_testing ())
X_train = X_train / 255.0
X_test = X_test / 255.0
n = X_train . shape [0]
d = X_train . shape [1] # 28 x 28

def find_partitions (X, MU):
    n = X. shape [0]
    partitions = {}
    for i in range (n):
        x = np. expand_dims (X[i, :], axis =0)
        temp1 = MU - x
        temp2 = np. linalg . norm (temp1 , axis =1)
        part = np. argmin ( temp2 )
        if part in partitions :
            partitions [ part ]. append (i)
        else :
            partitions [ part ] = [i]
    return partitions

def obj_funct (X, partitions , MU):
    cost = 0
    for part in partitions :
        mu = np. expand_dims (MU[part , :], axis =0)
        for j in partitions [ part ]:
            temp1 = mu - X[j, :]
            temp2 = np. linalg . norm ( temp1 ) ** 2
            cost += temp2
    return cost

def k_means (X, k):
    indices = np. random . choice ( list ( range (X. shape [0])), k, replace = False )
    #indices = random. sample ( range ( x . shape [ 1 ] ) , 1)
    MU = X[ indices , :]
    prev_MU = np. zeros ((k,d))
    delta = 1e-3
    obj_delta = 10
    objective_values = []
    itr = 0
    
    while np. amax (np. abs (MU - prev_MU )) > delta and ( len ( objective_values ) < 2 or abs (objective_values [-1] - objective_values [-2]) > obj_delta ):
        itr += 1
        prev_MU = np. copy (MU)
        partitions = find_partitions (X=X, MU=MU) # a dictionary that maps the part # ( the i in MU_i ) to the index ( the i in x_i )
        # Choose the new centroids
        for i in range (k):
            if i in partitions :
                MU[i, :] = 1/ len ( partitions [i]) * np.sum(X[ partitions [i], :], axis =0)
        objective_values . append ( obj_funct (X=X, partitions = partitions , MU=MU))
    plt . figure (1)
    plt . plot ( objective_values )
    plt . xlabel (" iterations ")
    plt . ylabel (" objective value ")
    plt . title ('Problem 3a')
    plt . show ()
    plt . figure ()
    #for i in range (k):
    #    plt . subplot (2,5,i+1)
    #    plt . imshow (np. reshape (MU[i, :], (28 , 28)))
    plt . show ()
    return MU , partitions

# Problem 6a)
k_means ( X_train , 10)

# Problem 6b)
train_errors = []
test_errors = []
k_vals = [2, 5, 10,20 ,40 ,80 ,160,640 ,1280 ] # 2560
for k in k_vals :
    print (" Running at k = ", k)
    MU_k , partitions_k = k_means ( X_train , k)
    train_error = 1/n * obj_funct ( X_train , partitions = partitions_k , MU= MU_k )
    partitions_test_k = find_partitions (X= X_test , MU= MU_k )
    test_error = 1/ X_test . shape [0] * obj_funct (X_test , partitions = partitions_test_k ,MU= MU_k )
    train_errors . append ( train_error )
    test_errors . append ( test_error )
    
#plt . figure (3)
plt . title ('Problem 3b: Error versus k-means)')
plt . plot ( k_vals , train_errors )
plt . plot ( k_vals , test_errors )
plt . xlabel (" k values ")
plt . ylabel (" Errors ")
plt . legend (['Train ', 'Test '])
plt . show ()