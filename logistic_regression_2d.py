
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42, cluster_std=1.2)


plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='1')
plt.legend()
plt.title("2D dataset")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def compute_cost(X, y, w, b):
    m, n = X.shape
    cost = 0
    
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)

    return cost / m

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        err = f_wb - y[i]
        
        for j in range(n):
            dj_dw[j] += err * X[i, j]

        dj_db += err

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        if i % max(1, num_iters // 10) == 0:
            cost = compute_cost(X, y, w, b)
            J_history.append(cost)
            print(f"Iteration {i:4}: Cost {cost:8.4f}")

    return w, b, J_history

def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)
    
    
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb >= 0.5 else 0
    return p


m, n = X.shape
initial_w = np.zeros(n)
initial_b = 0
iterations = 2000
alpha = 0.05

w, b, J_hist = gradient_descent(X, y, initial_w, initial_b, alpha, iterations)
print("w, b found by gradient descent:", w, b)


predictions = predict(X, w, b)
accuracy = np.mean(predictions == y) * 100
print(f"Training Accuracy: {accuracy:.2f}%")


plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='1')


x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
y_values = -(w[0] * x_values + b) / w[1]
plt.plot(x_values, y_values, label='decision boundary', color='black')

plt.legend()
plt.title("logistic regression boundary")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()
