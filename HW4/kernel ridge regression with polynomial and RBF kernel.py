import numpy as np
import matplotlib . pyplot as plt
from mpl_toolkits . mplot3d import Axes3D

def data_gen (n): # n: number of data points
    f = lambda x: 4  *np. sin (np.pi*x)*np. cos(6*np.pi* x ** 2)
    x = np. random . uniform (0, 1, n)
    err = np. random . normal ( size =(n, ))
    y = f(x) + err
    return (x,y) # x: input data,  y: input labels 

def find_K ( kernel_funct , x, n):
    return np. fromfunction ( lambda i, j: kernel_funct (x[i], x[j]), shape =(n,n), dtype = int )

def kernel_ridge_regress (x, y, k_f , lam , n):
    K = find_K ( kernel_funct =k_f , x=x, n=n)
    alpha_hat = np. linalg . solve (K + lam * np. eye (n), y)
    f = lambda z: np.sum(np. dot ( alpha_hat , k_f(z, x)))
    return f

# Part a)
def error_loo_tunning (x, y, kernel_funct , lam_val , hyperparam_val , n):
    errors = np. empty (( len( lam_val ), len ( hyperparam_val )))
    for row , lam in enumerate ( lam_val ):
        for col , hyperparam in enumerate ( hyperparam_val ):
            error_loo = 0
            for j in range (n):
                x_val = x[j]
                y_val = y[j]
                x_train = np. concatenate ((x[:j], x[j+1:]))
                y_train = np. concatenate ((y[:j], y[j+1:]))
                k_f = lambda x, xprime : kernel_funct (x, xprime , hyperparam )
                f = kernel_ridge_regress (x= x_train , y= y_train , k_f=k_f , lam =lam , n=n-1)
                error_loo += ( y_val - f( x_val )) ** 2
            error_loo = error_loo / n
            errors [row , col] = error_loo           
    lam_ind , h_ind = np. unravel_index (np. argmin (errors , axis = None ), errors . shape )
    return ( lam_val [ lam_ind ], hyperparam_val [ h_ind ]) # best parameter choice

def find_f_h (x,y,k_f , hyperparameter , lam , n, num_points = 1000 ):
    f_hat = kernel_ridge_regress (x=x,y=y, lam =lam , k_f= lambda x, xprime : k_f(x, xprime ,
    hyperparameter ), n=n)
    x_fine = np. linspace (0,1, num_points )
    y_kernel = [ f_hat (xi) for xi in x_fine ]
    return x_fine , y_kernel

def part_b (x,y, k_f , lam , hyperparameter , n, kernel_name ):
    plt . figure ()
    f_true = lambda x: 4 * np. sin (np.pi * x) * np.cos (6 * np.pi * x ** 2)
    x_fine , y_kernel = find_f_h (x=x,y=y,k_f=k_f , hyperparameter = hyperparameter , lam =lam , n=n)
    plt . ylim(-6, 6)    
    plt . plot (x, y, 'ko ')
    plt . plot ( x_fine , f_true ( x_fine ))
    plt . plot ( x_fine , y_kernel )
    plt . legend (['Data ', 'True ', 'Kernel Ridge Regression '])
    plt . ylabel ('y')
    plt . xlabel ('x')
    plt . title ('Problem 2b: Kernel Ridge Regression (' + kernel_name + ')')
    plt . show ()
    

def part_c (x,y, k_f , hyperparameter , lam , n, num_points =1000 , B= 300 ):
    fhat = np. zeros ((B, num_points ))
    for b in range (B):
        #print (b) # sample from the data with replacement
        indices = np. random . choice ( list ( range (n)), n, replace = True )
        x_b = x[ indices ]
        y_b = y[ indices ]
        _ , fhat [b, :] = find_f_h (x=x_b , y=y_b , k_f=k_f , hyperparameter = hyperparameter , lam =lam , n=n, num_points = num_points )
    percent_5 = np. percentile (fhat , q=5, axis =0)
    percent_95 = np. percentile (fhat , q=95 , axis =0)
    f_true = lambda x: 4 * np. sin (np.pi * x) * np.cos (6 * np.pi * x ** 2)
    x_fine , y_kernel = find_f_h (x,y,k_f , hyperparameter , lam , n)
    
    plt . figure ()
    plt . ylim(-6, 6)
    plt . plot (x, y, 'ko ')
    plt . plot ( x_fine , f_true ( x_fine ))
    plt . plot ( x_fine , y_kernel )
    plt . fill_between(x_fine, percent_5, percent_95, alpha=0.2)
    #plt . plot ( x_fine , percent_5 , 'r-')
    #plt . plot ( x_fine , percent_95 , 'b-')
    plt . legend (['data ', 'True ', 'f_hat '])
    plt . title ('Problem 2c ')
    plt . ylabel ('y')
    plt . xlabel ('x')
    plt . show ()

def ten_fold_CV (x, y, kernel_funct , lam_val , hyperparam_val , n):
    errors = np. empty (( len( lam_val ), len ( hyperparam_val )))
    for row , lam in enumerate ( lam_val ):
        for col , hyperparam in enumerate ( hyperparam_val ):
            error_CV = 0
            for j in range (10):
                start_val = j* (n // 10)
                end_val = (j+1) * (n // 10)
                x_val = x[ start_val : end_val ]
                y_val = y[ start_val : end_val ]
                x_train = np. concatenate ((x[: start_val ], x[ end_val :]))
                y_train = np. concatenate ((y[: start_val ], y[ end_val :]))
                k_f = lambda x, xprime : kernel_funct (x, xprime , hyperparam )
                f = kernel_ridge_regress (x= x_train , y= y_train , k_f=k_f , lam =lam , n=(n - n // 10) )
                error_CV += 1/ x_val . size * np. sum (np. square (( y_val - np. asarray ([f(x) for x in x_val ]))))
                
            errors [row , col] = error_CV

    lam_ind , h_ind = np. unravel_index (np. argmin (errors , axis = None ), errors . shape )
    return ( lam_val [ lam_ind ], hyperparam_val [ h_ind ])

############################################################################################
# parts b ,c)
n = 30
x, y = data_gen (n)
lam_val = [10 ** k for k in np. linspace (-6, 1, 25)] # all lambda values of interest
d_vals = list ( range (1, 11 , 1))
k_f_poly = lambda x, xprime , d: np. power ((1 + x * xprime ), d)
best_lam , best_d = error_loo_tunning (x=x, y=y, kernel_funct =k_f_poly , lam_val = lam_val , hyperparam_val =d_vals , n=n)
print (" Polynomial Kernel Function \n")
print (" best lambda : ", best_lam , " best d: ", best_d )

# Polynomial Kernel
part_b (x=x, y=y, k_f=k_f_poly , lam = best_lam , hyperparameter = best_d , n=n, kernel_name ='poly ')
part_c (x=x, y=y, k_f=k_f_poly , lam = best_lam , hyperparameter = best_d , n=n, num_points =1000 , B=300)

# RBF Kernel
gamma_ballpark = 1 / np. median ([(x[i] - x[j]) ** 2 for i in range (n) for j in range (n)])
print ('gamma ballpark : ', gamma_ballpark ) # find a ballpark estimate for gamma .
gamma_vals = np. linspace (1, 25 , 20)
k_f_rbf = lambda x, xprime , gamma : np.exp(- gamma * np. power (x - xprime , 2))
best_lam , best_gamma = error_loo_tunning (x=x, y=y, kernel_funct = k_f_rbf , lam_val = lam_val , hyperparam_val = gamma_vals , n=n)
print ("\n\ nRBF Kernel Function \n")
print (" best lambda : ", best_lam , " best gamma : ", best_gamma )
part_b (x=x, y=y, k_f=k_f_rbf , lam = best_lam , hyperparameter = best_gamma , n=n, kernel_name ='rbf ')
part_c (x=x, y=y, k_f=k_f_rbf , hyperparameter = best_gamma , lam = best_lam , n=n, num_points =1000 , B= 300 )

############################################################################################
# part d)
# Polynomial Kernel
n = 300
x, y = data_gen (n)
lam_val = [10 ** k for k in np. linspace (-6, 1, 20)] # all lambda values of interest
d_vals = list ( range (1, 21 , 1))
k_f_poly = lambda x, xprime , d: np. power ((1 + x * xprime ), d)
best_lam_poly , best_d = ten_fold_CV (x=x, y=y, kernel_funct =k_f_poly , lam_val = lam_val , hyperparam_val =d_vals , n=n)
print (" Polynomial Kernel Function \n")
print (" best lambda : ", best_lam_poly , " best d: ", best_d )
part_b (x=x, y=y, k_f=k_f_poly , lam = best_lam_poly , hyperparameter = best_d , n=n, kernel_name ='poly ')
part_c (x=x, y=y, k_f=k_f_poly , lam = best_lam_poly , hyperparameter = best_d , n=n, num_points =1000 , B= 300 )

# RBF Kernel
gamma_vals = np. linspace (1, 25 , 20)
k_f_rbf = lambda x, xprime , gamma : np.exp(- gamma * np. power (x - xprime , 2))
best_lam_rbf , best_gamma = ten_fold_CV (x=x, y=y, kernel_funct = k_f_rbf , lam_val = lam_val , hyperparam_val = gamma_vals , n=n)
print ("\n\ nRBF Kernel Function \n")
print (" best lambda : ", best_lam_rbf , " best gamma : ", best_gamma )
part_b (x=x, y=y, k_f=k_f_rbf , lam = best_lam_rbf , hyperparameter = best_gamma , n=n, kernel_name ='rbf')
part_c (x=x, y=y, k_f=k_f_rbf , hyperparameter = best_gamma , lam = best_lam_rbf , n=n, num_points =1000 ,B= 300 )

########################################################################
# part e)
m = 1000
xm , ym = data_gen (m)
# reuse old data to generate f_hats
f_hat_poly = kernel_ridge_regress (x=x,y=y, lam = best_lam_poly , k_f= lambda x, xprime : k_f_poly (x, xprime , best_d ), n=n)
f_hat_rbf = kernel_ridge_regress (x=x,y=y, lam = best_lam_rbf , k_f= lambda x, xprime : k_f_rbf (x, xprime , best_gamma ), n=n)
        
B = 300
bootstrap_val = np. zeros ((B, ))
for b in range (B):
    indices = np. random . choice ( list ( range (m)), m, replace = True )
    x_b = xm[ indices ]
    y_b = ym[ indices ]
    poly_err = np. square ( y_b - np. asarray ([ f_hat_poly (x) for x in x_b]))
    rbf_err = np. square ( y_b - np. asarray ([ f_hat_rbf (x) for x in x_b ]))
    bootstrap_val [b] = 1/m * np. sum ( poly_err - rbf_err )
        
percent_95 = np. percentile ( bootstrap_val , q=95)
percent_5 = np. percentile ( bootstrap_val , q=5)
print ('95th percentile : ', str ( percent_95 ))
print ('5th Percentile : ', str ( percent_5 ))