import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

digits = load_digits()
X, y = digits.data, digits.target
m, n = X.shape

# Visualize 10 sample digits
plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].reshape(8, 8), cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg

def gradient(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad
def one_vs_all(X, y, num_labels, lambda_):
    m, n = X.shape
    X = np.c_[np.ones(m), X]  # Add bias term
    all_theta = np.zeros((num_labels, n + 1))

    for i in range(num_labels):
        y_i = (y == i).astype(int)
        initial_theta = np.zeros(n + 1)
        result = minimize(fun=cost_function,
                          x0=initial_theta,
                          args=(X, y_i, lambda_),
                          method='TNC',
                          jac=gradient)
        all_theta[i] = result.x

    return all_theta
def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    probs = sigmoid(X @ all_theta.T)
    return np.argmax(probs, axis=1)

num_labels = 10
lambda_ = 0.1
all_theta = one_vs_all(X, y, num_labels, lambda_)
predictions = predict_one_vs_all(all_theta, X)
accuracy = accuracy_score(y, predictions)

print(f"Training Accuracy (One-vs-All): {accuracy * 100:.2f}%")

clf = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=500)
clf.fit(X, y)
sklearn_accuracy = clf.score(X, y)
print(f"Training Accuracy (scikit-learn): {sklearn_accuracy * 100:.2f}%")
