import numpy as np
import matplotlib.pyplot as plt

def f_m(x): #true function
    return 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)

n = 256
sigma = 1
eps = np.random.normal(0, sigma, n) #normal distribution (mean, variance, size) 
X = np.arange(0,n)/n #X = np.arange(1,n+1)/n 
Y = f_m(X) + eps

def f_m_hat(x, Y, m):#estimate
    n = Y.shape[0] #get Y first dimension length
    c = np.array([np.mean(Y[j*m:(j+1)*m]) for j in range(n//m)]) # [np.mean(Y[(j-1+1)*m+1:j+1*m])
    x_interval = x/(m/n) #x_interval_idx = x/(m/n) - 0.001 #avoid over 256
    x_interval = x_interval.astype('int') #Compute the interval idx (change data type to int)1
    return c[x_interval]

def mse(x, y):
    return np.mean((x-y)**2)

def bias_sq(X, f, m):#b
    n = X.shape[0]
    f_arr = f(X)
    fbar = np.array([np.mean(f_arr[j*m:(j+1)*m]) for j in range(n//m)])
    bias_sq = []
    for j in range(n//m):
        for i in range(j*m, (j+1)*m):
            bias_sq.append((fbar[j] - f_arr[i])**2)
    return np.mean(bias_sq)

def var_sq(sigma, m):#c
    return sigma**2/m

#result
m = [1,2,4,8,16,32] 
empirical_mse_arr = np.array([mse(f_m_hat(X, Y, mi),f_m(X)) for mi in m])
bias_sq_arr = np.array([bias_sq(X, f_m, mi) for mi in m]) #b
var_sq_arr = np.array([var_sq(sigma, mi) for mi in m]) #c
avg_error_arr = bias_sq_arr + var_sq_arr

#plot
plt.figure(figsize = (20, 15))
plt.plot(m, empirical_mse_arr, '-o', label = 'average_empirical_error') 
plt.plot(m, bias_sq_arr, '-o', label = 'average_bias_squared')
plt.plot(m, var_sq_arr, '-o', label = 'average_variance_squared') 
plt.plot(m, avg_error_arr, '-o', label = 'average_error')
plt.legend() 
plt.title('hw1 3.d')
plt.xlabel('m')
plt.ylabel('value')
plt.show()
    