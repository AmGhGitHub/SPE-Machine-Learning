import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor, LinearRegression

mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

np.random.seed(42)


def learning_schedule(t, t0=50, t1=10_000):
    # t0 and t1 are learning schedule hyper-parameters
    return t0 / (t + t1)


def design_matrix(x):
    x = x.copy().flatten()
    X = np.c_[np.ones((x.shape[0], 1)), x.reshape(-1, 1)]  # add the "1" column vector to vector x
    return X


# batch gradient descent
def batch_gd(x, y, max_iter=15_000, eta0=0.015):
    x = design_matrix(x)
    y = y.copy().reshape(-1, 1)
    m = x.shape[0]
    beta_path = np.zeros((max_iter, 2))

    beta = np.random.randn(2, 1)  # random initialization
    for itr in range(max_iter):
        delta_beta = (2.0 / m) * x.T.dot(x.dot(beta) - y)
        beta -= eta0 * delta_beta
        beta_path[itr, 0] = beta[0]
        beta_path[itr, 1] = beta[1]

    return beta, beta_path


# Stochastic Gradient Descent
def stochastic_gd(x, y, n_epochs=100):
    x = design_matrix(x)
    y = y.copy().reshape(-1, 1)
    m = x.shape[0]
    beta_path = np.zeros((m * n_epochs, 2))

    beta = np.random.randn(2, 1)  # random initialization
    counter = 0
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = x[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]
            delta_beta = 2.0 * xi.T.dot(xi.dot(beta) - yi)
            eta0 = learning_schedule(epoch * m + i)
            beta -= eta0 * delta_beta
            beta_path[counter, 0] = beta[0]
            beta_path[counter, 1] = beta[1]
            counter += 1

    return beta, beta_path

# mini-batch Gradient Descent
def mini_batch_gd(x, y, max_iter=15_000, batch_size=20):
    x = design_matrix(x)
    y = y.copy().reshape(-1, 1)
    m = x.shape[0]

    beta = np.random.randn(2, 1)  # random initialization
    t, t0, t1 = 0, 10, 1_000
    beta_path = np.zeros((max_iter, 2))
    for epoch in range(max_iter):
        shuffled_indices = np.random.permutation(m)
        x_shuffled = x[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, batch_size):
            t += 2
            xi = x_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]
            delta_beta = (2.0 / batch_size) * xi.T.dot(xi.dot(beta) - yi)
            eta0 = learning_schedule(t, t0, t1)
            beta -= eta0 * delta_beta
        beta_path[epoch, 0] = beta[0]
        beta_path[epoch, 1] = beta[1]

    return beta, beta_path


df = pd.read_csv('poro_vs_sw.csv')
x = df['porosity-%'].values
y = df['sw-%'].values

# Apply Batch Gradient Descent
b_bgd, bp_bgd = batch_gd(x, y, max_iter=15_000, eta0=0.015)
size_bgd = bp_bgd.shape[0]

# Apply Stochastic Gradient Descent
b_sgd, bp_sgd = stochastic_gd(x, y, n_epochs=100)

# Apply Mini-batch Gradient Descent
b_mbgd, bp_mbgd = mini_batch_gd(x, y, max_iter=15_000, batch_size=20)

lr = LinearRegression().fit(x.reshape(-1, 1), y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
step_size = 100
ax.plot(bp_bgd[::100, 0], bp_bgd[::100, 1], "rs", label="Stochastic GD")
ax.plot(bp_sgd[::100, 0], bp_sgd[::100, 1], "g+", label="Mini-batch GD")
ax.plot(bp_mbgd[::100, 0], bp_mbgd[::100, 1], "bo", label="Batch GD")
ax.plot(lr.intercept_, lr.coef_, "kd", markersize=8, label="Normal Equation")
ax.legend(loc="upper left", fontsize=16)
ax.set_xlabel(r"$\beta_0$", fontsize=20)
ax.set_ylabel(r"$\beta_1$", fontsize=20)
plt.tight_layout()
plt.show()
