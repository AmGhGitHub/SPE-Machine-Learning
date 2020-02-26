# code 8
import numpy as np
from sklearn.linear_model import LinearRegression

# from this point on we use the typical variables names for machine learning
x_train = np.array([[46, 5], [48, 4], [50, 3], [54, 5], [58.5, 3]])
y_train = np.array([70, 90.2, 130, 175.1, 180])

x_test = np.array([[48, 5], [49, 3], [51, 5], [56, 5], [52, 3]])
y_test = np.array([110, 85, 150, 180, 110])

ml_model = LinearRegression()
ml_model.fit(x_train, y_train)
beta = np.concatenate([np.array([ml_model.intercept_]), ml_model.coef_])
print(f"SKL:\n \u03B20 = {beta[0]:.4f}; \u03B21 = {beta[1]:.4f}; \u03B22 = {beta[2]:.4f}")

r2 = ml_model.score(x_test, y_test)
print(f"SKL:\n Model's R2 = {100 * r2:.2f}%")
