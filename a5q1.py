import numpy as np
import matplotlib.pyplot as plt

#X=np.array([6.1101,5.5277,8.5186])
#y=np.array([17.592,9.1302,13.662])
#m=len(y)

data=np.loadtxt('ex2data1.txt',delimiter=',')
X=data[:,0]
y=data[:,1]
m=len(y)
#"""
def z_normalize(X):
    mu=np.mean(X)
    sigma=np.std(X)
    X_norm=(X-mu)/sigma
    return X_norm

X=z_normalize(X)
y=z_normalize(y)

#"""
def compute_cost(X,y,w,b):
    cost=0
    for i in range(m):
        f_wb=np.dot(w,X[i])+b
        cost += (f_wb-y[i])**2
    cost=cost/(2*m)
    return cost
def gradient(X,y,w,b):
    dw=0
    db=0
    for i in range(m):
        f_wb=np.dot(w,X[i])+b
        dw += (f_wb-y[i])*X[i]
        db += (f_wb-y[i])
    dw=(1/m)*dw
    db=(1/m)*db
    return dw,db
def gradient_descent(X,y,w,b,alpha,iterations):
    for i in range(iterations):
        dw,db=gradient(X,y,w,b)
        w -= alpha*dw
        b -= alpha*db
    return w,b

w,b=gradient_descent(X,y,0,0,0.01,1000)

print(w)
print(b)
           
plt.scatter(X,y)
plt.plot(X,w*X+b)
plt.show()           
