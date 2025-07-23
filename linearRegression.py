import numpy as np
import matplotlib.pyplot as plt
import copy
import math

#dataset
y_train = np.array([10.0, 20.0, 30.0, 38.0, 40.0, 44.0, 47.0, 50.0, 52.0, 54.0, 55.0])
x_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

print("x_train:", x_train)
print("\ny_train:", y_train)
print('\nnum of training examples:', len(x_train))

plt.scatter(x_train, y_train, marker='x', c='r')
plt.show()

#polynomial model
def prepare_polynomial_features(x, degree):
    m = x.shape[0]
    X_poly = np.zeros((m, degree))
    
    for i in range(degree):
        X_poly[:, i] = x ** (i + 1)
    
    #normalization
    mean = np.mean(X_poly, axis=0)
    std = np.std(X_poly, axis=0)
    return (X_poly - mean) / std




#using a 3rd degree polynomial
d = 3
X_poly = prepare_polynomial_features(x_train, degree=d)

#cost for multiple variables model
def compute_cost_multi(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost += (f_wb - y[i])**2
    return cost / (2*m)

#gradient for multiple variables model
def compute_gradient_multi(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        err = f_wb - y[i]
        dj_dw += err * X[i]
        dj_db += err
    return dj_dw/m, dj_db/m

#gradient descent => a = a - learningRate*derivative
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % math.ceil(num_iters/10) == 0:
            cost = cost_function(X, y, w, b)
            J_history.append(cost)
            print(f"Iteration {i:4}: Cost {cost:8.2f}")
    return w, b, J_history

#initial
initial_w = np.zeros(d)
initial_b = 0.
iterations = 10000
alpha = 0.001

#traning
w, b, J_hist = gradient_descent(X_poly, y_train, initial_w, initial_b,
                                compute_cost_multi, compute_gradient_multi,
                                alpha, iterations)
print("w, b found by gradient descent:", w, b)

#prediction
m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = np.dot(X_poly[i], w) + b

plt.plot(x_train, predicted, c="b", label="polynomial fit")
plt.scatter(x_train, y_train, marker='x', c='r', label="training Data")
plt.title("polynomial regression fit")
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()


