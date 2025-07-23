
import numpy as np
import matplotlib.pyplot as plt

#dataset
x_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Binary labels

#adding bias term separately
X_train = x_train.reshape(-1, 1)

print("x_train:", x_train)
print("y_train:", y_train)
print("num of training examples:", len(x_train))

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#cost function
def compute_cost(X, y, w, b):
    m, n = X.shape
    cost = 0
    
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        cost += -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)

    return cost / m

#gradient
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


#gradient descent
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

#prediction
def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)
    
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb >= 0.5 else 0

    return p

#initialize parameters
w_init = np.zeros(X_train.shape[1])
b_init = 0
iterations = 2000
alpha = 0.05

#training
w, b, J_hist = gradient_descent(X_train, y_train, w_init, b_init, alpha, iterations)
print("w, b found by gradient descent:", w, b)

#predictions
predictions = predict(X_train, w, b)
print("Train Accuracy: {:.2f}%".format(np.mean(predictions == y_train) * 100))


x_values = np.linspace(0, 10, 100)
y_values = sigmoid(w[0]*x_values + b)
plt.plot(x_values, y_values, label="sigmoid")
plt.scatter(x_train, y_train, marker='x', c='r', label="data")
plt.title("logistic regression fit")
plt.xlabel("x")
plt.ylabel("prob")
plt.legend()
plt.show()
