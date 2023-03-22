import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# initialize variables
n = 500
d = 1000
k = 100
sigma = 1

x = np.random.normal(0, 1, (d, n))
w = np.zeros(d)
w[0:k] = np.arange(1, k + 1) / k

# as defined in problem
y = np.matmul(np.transpose(w), x) + np.random.normal(0, 1, n)


# to find the maximum lambda that gives all zero in w
ypre = y - np.sum(y) / n
xpre = np.matmul(x, ypre)
xsum = 2 * np.absolute(xpre)
lambdamax = np.max(xsum)

# initialize variables
deltalim = 0.01
maxfeatures = 1*d
nonzero = 0
lambdas = []
nonzeros = []
FDR = []
TPR = []

# Keep checking smaller lambdas until at least 900 elements in w are nonzero
a = 2 * np.sum(np.square(x), axis=1)
while nonzero < maxfeatures:
    lambdas.append(lambdamax)
    print('lambdamax', lambdamax)
    w = np.zeros(d)

    # gradient descent
    maxdelta = float('inf')
    while maxdelta > deltalim:
        # a and b can be calculated outside of the for loop
        b = np.sum(y - np.matmul(w, x)) / n
        oldw = w.copy()
        for j in range(d):
            # calculate ck
            tempw = np.delete(w, j)
            tempx = np.delete(x, j, 0)
            ck = 2 * np.dot(np.transpose(x[j, :]), y - (b + np.matmul(tempw, tempx)))
            # determine wk
            if ck < -lambdamax:
                w[j] = (ck + lambdamax) / a[j]
            elif ck > lambdamax:
                w[j] = (ck - lambdamax) / a[j]
            else:
                w[j] = 0

        # find difference, set variables for conditions
        delta = np.absolute(oldw - w)
        maxdelta = np.max(delta)

        # sanity check to make sure objective gets smaller each iteration
        tempb = np.sum(y - np.matmul(w, x)) / n
        err = np.sum(np.square(np.matmul(w, x)+tempb-y)) + lambdamax*np.sum(np.absolute(w))
        print('error: ', err)

    # save number of nonzero elements and set new lambda
    nonzero = np.count_nonzero(w)
    print('nonzero:', nonzero)
    nonzeros.append(nonzero)
    lambdamax /= 1.5

    # calculate TPR and FDR
    TPR.append(np.count_nonzero(w[0: k]) / k)
    if nonzero == 0:
        FDR.append(0)
    else:
        FDR.append(np.count_nonzero(w[k: d]) / nonzero)

# plot1
plt.figure(figsize=(15, 10))
plt.title('plot7.1')
plt.plot(lambdas, nonzeros)
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Non-zeros')

# plot2
plt.figure(figsize=(15, 10))
plt.title('plot7.2')
plt.plot(FDR, TPR)
plt.xlabel('FDR')
plt.ylabel('TPR')