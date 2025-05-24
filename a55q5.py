import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

def generate_data():
    np.random.seed(0)
    X = np.linspace(0, 10, 100)
    y = X * np.sin(X) + np.random.randn(*X.shape) * 2
    return X.reshape(-1, 1), y

X, y = generate_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

def prepare_poly_features(X, degree):
    poly = PolynomialFeatures(degree, include_bias=False)
    return poly.fit_transform(X), poly

def cost_function_reg(theta, X, y, lambda_):
    m = len(y)
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    predictions = X @ theta
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    reg = (lambda_ / (2*m)) * np.sum(theta[1:] ** 2)
    return cost + reg

def gradient_reg(theta, X, y, lambda_):
    m = len(y)
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    error = X @ theta - y
    grad = (X.T @ error) / m
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad.flatten()

def train_linear_reg(X, y, lambda_):
    initial_theta = np.zeros(X.shape[1])
    result = minimize(fun=cost_function_reg,
                      x0=initial_theta,
                      args=(X, y, lambda_),
                      method='TNC',
                      jac=gradient_reg)
    return result.x

def learning_curve(X_train, y_train, X_val, y_val, lambda_):
    m = X_train.shape[0]
    error_train = []
    error_val = []

    for i in range(1, m + 1):
        theta = train_linear_reg(X_train[:i], y_train[:i], lambda_)
        error_train.append(cost_function_reg(theta, X_train[:i], y_train[:i], 0))
        error_val.append(cost_function_reg(theta, X_val, y_val, 0))

    return error_train, error_val

degree = 5
lambda_ = 1.0

X_poly_train, poly = prepare_poly_features(X_train, degree)
X_poly_val = poly.transform(X_val)

theta = train_linear_reg(X_poly_train, y_train, lambda_)
print("Learned parameters (theta):", theta)

# Plot polynomial fit
X_fit = np.linspace(0, 10, 100).reshape(-1, 1)
X_fit_poly = poly.transform(X_fit)
y_fit = X_fit_poly @ theta

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, c='blue', label='Training data')
plt.scatter(X_val, y_val, c='orange', label='Validation data')
plt.plot(X_fit, y_fit, c='green', linewidth=2, label=f'Polynomial Fit (deg={degree}, λ={lambda_})')
plt.title('Polynomial Regression Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

error_train, error_val = learning_curve(X_poly_train, y_train, X_poly_val, y_val, lambda_)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(error_train) + 1), error_train, label='Train Error')
plt.plot(range(1, len(error_val) + 1), error_val, label='Validation Error')
plt.title('Learning Curve')
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

lambdas = [0, 0.01, 1, 100]
for l in lambdas:
    theta = train_linear_reg(X_poly_train, y_train, l)
    y_fit = poly.transform(X_fit) @ theta

    plt.plot(X_fit, y_fit, label=f'λ = {l}')

plt.scatter(X_train, y_train, c='blue', label='Training Data')
plt.title('Polynomial Regression for Different λ')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
