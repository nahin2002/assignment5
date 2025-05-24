import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize


data=np.loadtxt('ex2data1.txt',delimiter=',')
X1=data[:,0]
X2=data[:,1]
y=data[:,2]
X=np.vstack((X1,X2)).T

def z_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

X, mu, sigma = z_normalize(X)  # Normalize features

print(X)

def map_features(X1,X2,degree=6):
    poly=PolynomialFeatures(degree)
    return poly.fit_transform(np.column_stack((X1,X2)))

X_mapped=map_features(X1,X2)
print(X_mapped)

"""
def generate_data():
    np.random.seed(0)
    m = 100
    X = np.random.randn(m, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 + 0.3*np.random.randn(m) < 1.5).astype(int)
    return X, y

X, y = generate_data()
"""

def map_features(X1, X2, degree=6):
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(np.column_stack((X1, X2)))

X_mapped = map_features(X[:, 0], X[:, 1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg

def gradient_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    error = h - y
    grad = (X.T @ error) / m
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad
def train_logistic_reg(X, y, lambda_):
    initial_theta = np.zeros(X.shape[1])
    res = minimize(fun=cost_function_reg,
                   x0=initial_theta,
                   args=(X, y, lambda_),
                   method='TNC',
                   jac=gradient_reg)
    return res.x

def plot_decision_boundary(theta, X, y, lambda_):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

    u = np.linspace(-2, 2, 100)
    v = np.linspace(-2, 2, 100)
    z = np.zeros((u.size, v.size))
    
    for i in range(u.size):
        for j in range(v.size):
            mapped = map_features(np.array([u[i]]), np.array([v[j]]))
            z[i, j] = mapped @ theta

    z = z.T
    plt.contour(u, v, z, levels=[0], linewidths=2, colors='green')
    plt.title(f"Decision Boundary (λ = {lambda_})")
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")
    plt.grid(True)
    plt.show()
for lambda_ in [0, 1, 100]:
    theta = train_logistic_reg(X_mapped, y, lambda_)
    
    plot_decision_boundary(theta, X, y, lambda_)
