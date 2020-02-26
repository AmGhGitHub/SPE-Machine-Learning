# code 6
import numpy as np
from sklearn.linear_model import LinearRegression

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# train data
actual_capex = np.array([[46], [48], [50], [54], [58.5]])
actual_npv = np.array([70, 90.2, 130, 175.1, 180])
# test data
test_capex = np.array([[48], [49], [51], [56], [52]])
test_npv = np.array([110, 85, 150, 180, 110])

sl_model = LinearRegression()
sl_model.fit(actual_capex, actual_npv)

predicted_test_npv = sl_model.predict(test_capex)

ss_total = ((test_npv - test_npv.mean()) ** 2).sum()
ss_res = ((test_npv - predicted_test_npv) ** 2).sum()
print(f"Formula\n {'R2'.translate(SUP)} = {1.0 - ss_res / ss_total:.5f}\n")
print(f"SKL:\n {'R2'.translate(SUP)} = {sl_model.score(test_capex, test_npv):.5f}")
