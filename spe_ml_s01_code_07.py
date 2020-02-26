# code 7
import numpy as np

# from this point on we use the typical variables names for machine learning
x_train = np.array([[46, 5], [48, 4], [50, 3], [54, 5], [58.5, 3]])
y_train = np.array([70, 90.2, 130, 175.1, 180])

x_test = np.array([[48, 5], [49, 3], [51, 5], [56, 5], [52, 3]])
y_test = np.array([110, 85, 150, 180, 110])

# d stands for design matrix
x_train_d = np.c_[np.ones((x_train.shape[0], 1)), x_train]
print(f"Design Matrix:\n {x_train_d}\n")

beta = np.dot(np.linalg.inv(np.dot(np.transpose(x_train_d), x_train_d)), np.dot(np.transpose(x_train_d), y_train))
print(f"Numpy 'inv':\n \u03B20 = {beta[0]:.4f}; \u03B21 = {beta[1]:.4f}; \u03B22 = {beta[2]:.4f}\n")

# NumPy also provides a least squares function that can solve the values of the
# parameters more compactly; Also, this is the way, SKL calculates the
beta = np.linalg.lstsq(x_train_d, y_train, rcond=-1)[0]
print(f"Numpy 'lstsq':\n \u03B20 = {beta[0]:.4f}; \u03B21 = {beta[1]:.4f}; \u03B22 = {beta[2]:.4f}\n")
