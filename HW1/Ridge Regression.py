import numpy as np
import matplotlib.pyplot as plt

train_n = 100
test_n = 1000
d = 100
x_train = np.random.normal(0,1, size=(train_n,d))
a_true = np.random.normal(0,1, size=(d,1))
y_train = x_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
x_test = np.random.normal(0,1, size=(test_n,d))
y_test = x_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))

lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

#Intial normalized error arrays with dimension of (lamdas, trails)
norm_training_error = np.zeros((len(lambdas), 30))
norm_test_error = np.zeros((len(lambdas), 30))

for count, value in enumerate(lambdas): 
    for y in range(30):
        w1 = np.linalg.inv(np.dot((x_train.T),x_train) + value*np.identity(d))
        w2 = np.dot((x_train.T),y_train)
        w_hat = np.dot(w1,w2)
        #norm
        temp_training_error = np.linalg.norm(np.dot(x_train,w_hat)-y_train) / np.linalg.norm(y_train)
        temp_test_error     = np.linalg.norm(np.dot(x_test,w_hat)-y_test) / np.linalg.norm(y_test)
        #storing error values for every value of lamda and trails
        norm_training_error[count, y] = temp_training_error
        norm_test_error[count, y] = temp_test_error
        
#calculating average along each row for all the trails
avg_trainig_errors = np.mean(norm_training_error, axis=1) #axis=1 working along the row
avg_test_errors = np.mean(norm_test_error, axis=1)

#plot
plt.figure(figsize = (12, 8))
plt.semilogx(lambdas, avg_trainig_errors,'-o', label = 'Training errors')
plt.semilogx(lambdas, avg_test_errors,'-o', label = 'Testing errors')
plt.legend()
plt.title('hw1-4')
plt.xlabel('lambdas')
plt.ylabel('Normalized Error')
plt.savefig('hw1-4.png')
plt.show()
