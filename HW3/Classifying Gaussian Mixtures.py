import matplotlib . pyplot as plt
import numpy as np

def part_a (mu , sig , n):
    L, Q = np.linalg .eig(sig)
    L = np. diag (L) # convert L to a matrix
    A = np. matmul (np. matmul (Q, np. sqrt (L)), Q.T)
    np.random.seed ( 1 )
    Z = np. random . randn (2, n)
    X = np. matmul (A, Z) + mu
    plt . figure (1)
    plt . plot (X[0, :], X[1, :], '^')
    plt . xlabel ('average model')
    plt . ylabel ('optimal line')
    plt . title ('Problem 4c ')
    plt . show ()
    return X

def part_b (X, n):
    mu_hat1 = 1/n * np. sum (X, axis =1)
    mu_hat = np. expand_dims (mu_hat1 , axis =1)
    mu_temp = X - mu_hat
    sig_hat = 1/(n-1) * np. sum ([np.outer(mu_temp[:, i], mu_temp[:, i]) for i in range (n)])
    L, Q = np. linalg . eig ( sig_hat )
    start = mu_hat
    end1 = mu_hat + np. expand_dims (np. sqrt (L[0]) * Q[:, 0], axis =1)
    end2 = mu_hat + np. expand_dims (np. sqrt (L[1]) * Q[:, 1], axis =1)
    points1 = np. hstack ((start , end1))
    points2 = np. hstack ((start , end2))
    plt . figure (1)
    plt . plot ( mu_hat [0], mu_hat [1], 'ro ')
    plt . plot ( points1 [0, :], points1 [1, :])
    plt . plot ( points2 [0, :], points2 [1, :])
    r = max (np. amax (np. abs ( points1 )), np. amax (np. abs ( points2 ))) + 1
    limits = [-r, r]
    return mu_hat , sig_hat , L, Q
                   
def part_c (X, mu_hat , L, Q):
    row1 = 1/np. sqrt (L[0]) * np. dot (Q[:, 0], X - mu_hat )
    row2 = 1/np. sqrt (L[1]) * np. dot (Q[:, 1], X - mu_hat )
    r = 1 + max (np. amax (np. abs ( row1 )), np. amax (np. abs ( row2 )))

                   

n = 1000
mu1 = np. asarray ([[-1], [1]])
sigma1 = np. asarray ([[1, 0], [0, 2]])
X1 = part_a (mu1 , sigma1 , n)                  
mu1_hat , sig_hat , L1 , Q1 = part_b (X1 , n)
part_c (X1 , mu1_hat , L1 , Q1)
                   
mu2 = np. asarray ([[1], [1]])
sigma2 = np. asarray ([[1, 0], [0, 2]])
X2 = part_a (mu2 , sigma2 , n)
mu2_hat , Sigma2_hat , L2 , Q2 = part_b (X2 , n)
part_c (X2 , mu2_hat , L2 , Q2)