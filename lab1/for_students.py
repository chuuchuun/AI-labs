import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
X_b = np.c_[np.ones((len(x_train), 1)), x_train]
theta_best = np.linalg.inv(X_b.T.dot(X_b)) @ X_b.T.dot(y_train)
print(f'Theta[0]:{theta_best[0]}, Theta[1]: {theta_best[1]}')

# TODO: calculate error
def find_mse(x, y, theta):
    X = np.c_[np.ones((len(x), 1)), x]
    m = len(X)
    predictions = X.dot(theta)
    mse = (1 / m) * np.sum((predictions - y) ** 2)
    return mse


print(f'MSE testing: {find_mse(x_test, y_test, theta_best)}')
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
z_x_train = (x_train - np.mean(x_train)) / np.std(x_train)
z_y_train = (y_train - np.mean(y_train)) / np.std(y_train)

# TODO: calculate theta using Batch Gradient Descent
theta = np.random.rand(2)
learning_rate = 0.01
z_b = np.c_[np.ones((len(z_x_train), 1)), z_x_train]
for i in range(1000):
    m = len(z_b)
    z_b_trans = z_b.T
    predictions_train = z_b.dot(theta)
    errors_train = predictions_train - z_y_train
    mse_grad_train = 2 / m * z_b_trans.dot(errors_train)
    theta = theta - learning_rate * mse_grad_train
    print(f'Theta[0]: {theta[0]}, Theta[1]: {theta[1]}')

# TODO: calculate error
z_x_test = (x_test - np.mean(x_train)) / np.std(x_train)
z_b_test = np.c_[np.ones((len(z_x_test), 1)), z_x_test]
z_y_test_predicted = z_b_test.dot(theta)
z_y_test = z_y_test_predicted * np.std(y_train) + np.mean(y_train)
z_m = len(z_x_test)
z_mse = (1 / z_m) * np.sum((z_y_test - y_test) ** 2)

print(f'MSE testing after standardization: {z_mse}')

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
