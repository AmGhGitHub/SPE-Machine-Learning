# code 5
import numpy as np
from sklearn.linear_model import LinearRegression

# SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

actual_capex = np.array([[46], [48], [50], [54], [58.5]])
actual_npv = np.array([70, 90.2, 130, 175.1, 180])

sl_model = LinearRegression()
sl_model.fit(actual_capex, actual_npv)

xy = np.c_[actual_capex, actual_npv].T
cov_xy = np.cov(xy)[0, 1]
var_x = np.cov(xy)[0, 0]
beta1 = cov_xy / var_x
beta0 = xy.mean(axis=1)[1] - beta1 * xy.mean(axis=1)[0]

print(f"Formula:\n \u03B20 = {beta0}\n \u03B21 = {beta1}\n")
print(f"SKL:\n \u03B20 = {sl_model.intercept_}\n \u03B21 = {sl_model.coef_[0]}\n")
