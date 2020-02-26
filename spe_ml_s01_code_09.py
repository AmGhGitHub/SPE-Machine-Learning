# code 9
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train = np.array([[6], [8], [10], [14], [18]])
y_train = np.array([7, 9, 13, 17.5, 18])
x_test = np.array([[6], [8], [11], [16]])
y_test = np.array([8, 12, 15, 18])

quadratic_featurizer = PolynomialFeatures(degree=2)
x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

quadratic_regressor = LinearRegression()
quadratic_regressor.fit(x_train_quadratic, y_train)
r_sq = quadratic_regressor.score(x_test_quadratic, y_test)
print(f"Quadratic Regression:\n R2 = {r_sq:.4f}")

