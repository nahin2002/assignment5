import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('ex2data1.txt',delimiter=',')
X=data[:,0:2]
x1=X[:,0]
x2=X[:,1]
y=data[:,2]
def z_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X, mu, sigma = z_normalize(X)  # Normalize features

def compute_cost(X, y, w, b):
    m = len(y)
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        cost += (f_wb - y[i]) ** 2
    return cost / (2 * m)

def gradient(X, y, w, b):
    m, n = X.shape
    dw = np.zeros(n)
    db = 0.0
    for i in range(m):
        err = np.dot(w, X[i]) + b - y[i]
        for j in range(n):
            dw[j] += err * X[i, j]
        db += err
    dw /= m
    db /= m
    return dw, db

def gradient_descent(X, y, w, b, alpha, iterations):
    cost_history=[]
    for i in range(iterations):
        dw, db = gradient(X, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            cost_history.append(cost)
            print(f"Iteration {i}: Cost = {cost}")    
    return w, b,cost_history


w_init = np.zeros(X.shape[1])
b_init = 0
alpha = 0.01
iterations = 400

w, b,cost_history = gradient_descent(X, y, w_init, b_init, alpha, iterations)

print("Weights:", w)
print("Bias:", b)
m=len(y)
negative=y==0
positive=y==1
y_pred=np.dot(X,w)+b

def plot_decision(w,b,mu,sigma):
    plt.figure(figsize=(10,6))

    plt.scatter(x1[positive],x2[positive],c='blue',marker='o',label='class 1')
    plt.scatter(x1[negative],x2[negative],c='red',marker='x',label='class 0')
   
    x1_min,x1_max=x1.min()-1,x1.max()+1
    x1_values=np.linspace(x1_min,x1_max,100)

    x2_values=[]

    for x1_val in x1_values:
        x1_norm=(x1_val-mu[0])/sigma[0]
        x2_norm=(0.5-w[0]*x1_norm-b)/w[1]
        x2_val=x2_norm*sigma[1]+mu[1]
        x2_values.append(x2_val)

    plt.plot(x1_values,x2_values,'g-',label='Decision Boundary (prediction =0.5)')    
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.grid(True)
    plt.show() 
    


# Plot the decision boundary
plot_decision(w, b, mu, sigma)

def plot_prediction_plane(X,y,w,b,mu,sigma):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    positive=y==1
    negative=y==0
    ax.scatter(x1[positive],x2[positive],y[positive],c='blue',marker='o',label='calss 1')
    ax.scatter(x1[negative],x2[negative],y[negative],c='red',marker='x',label='class 0')

    x1_min,x1_max=x1.min()-1,x1.max()+1
    x2_min,x2_max=x2.min()-1,x2.max()+1
    xx1,xx2=np.meshgrid(np.linspace(x1_min,x1_max,100),np.linspace(x2_min,x2_max,100))
    zz=np.zeros(xx1.shape)
    
    xx1_norm=(xx1-mu[0])/sigma[0]
    xx2_norm=(xx2-mu[1])/sigma[1]

    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            zz[i,j]=w[0]*xx1_norm[i,j]+w[1]*xx2_norm[i,j]+b
    ax.plot_surface(xx1,xx2,zz,alpha=0.3)
    plt.show()

plot_prediction_plane(X,y,w,b,mu,sigma)

w1=np.linspace(-0.5,0.5,100)
w2=np.linspace(-0.5,.5,100)
J_vals = np.zeros((len(w1), len(w2)))

# Compute cost over the grid
for i in range(len(w1)):
    for j in range(len(w2)):
        t = np.array([w1[i], w2[j]])
        J_vals[i, j] = compute_cost(X, y,t,0.58923)

# Transpose J_vals for correct plotting
J_vals = J_vals.T

# Contour plot
plt.figure(figsize=(8, 6))
CS = plt.contour(w1, w2, J_vals, levels=20, cmap='viridis')
plt.clabel(CS, inline=1, fontsize=8)
plt.plot(w[0], w[1], 'rx', markersize=10, linewidth=2, label='Optimal w')
plt.legend()

plt.xlabel('Theta 0 Feature 0 weight')
plt.ylabel('Theta 1 (Feature 1 weight)')
plt.title('Cost Function Contour Plot (Theta 2 fixed)')
plt.grid(True)
plt.show()
plt.show()

