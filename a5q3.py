import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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

def sigmoid(z):
    return(1/(1+np.exp(-z)))
def compute_cost(X, y, w, b):
    m = len(y)
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        g=sigmoid(f_wb)
        cost += -y[i]*np.log(g) -(1-y[i])*np.log(1-g)
        
    return cost/m

def gradient(X, y, w, b):
    m, n = X.shape
    dw = np.zeros(n)
    db = 0.0
    for i in range(m):
        f_wb=np.dot(w, X[i]) + b
        err = sigmoid(f_wb) - y[i]
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
iterations = 1000

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


w1=np.linspace(-1.5,1.5,100)
w2=np.linspace(-1.5,1.5,100)
J_vals = np.zeros((len(w1), len(w2)))

# Compute cost over the grid
for i in range(len(w1)):
    for j in range(len(w2)):
        t = np.array([w1[i], w2[j]])
        J_vals[i, j] = compute_cost(X, y,t,b)

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


def predict(X, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    return probs >= 0.5

preds = predict(X, w, b)
accuracy = np.mean(preds == y) * 100
print(f"Training Accuracy: {accuracy:.2f}%")

print('Train model: ')

lr_model=LogisticRegression()
lr_model.fit(X,y)
print('Train Accuracy: ',lr_model.score(X,y)*100)

