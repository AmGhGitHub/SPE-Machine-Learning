import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)

df_rock = pd.read_csv('data/rock_type.csv', header=[0])
feature_name = 'Porosity (%)'
x = df_rock[feature_name].values
x = x.reshape(-1, 1)
# convert categorical data to numerical
y = (df_rock['Rock Type'] == 'SS').astype(np.int)

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.1, random_state=42)
log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(x_train, y_train)
beta0 = log_reg.intercept_[0]
beta1 = log_reg.coef_[0, 0]
print(f"Logistic Regression's Coefficients: \u03B20 = {beta0:0.3f} & \u03B21 = {beta1:.3f}")
print("\n**Probability Estimation**")
print(f"Analytical solution @ x=6: {1 / (1 + np.exp(-(beta0 + beta1 * 6.0))):.3f}")
print(f"Logistic Regression proba() method @ x=6: {log_reg.predict_proba([[6]])[0, 1]:.3f}\n\n")
x_prediction = np.linspace(2, 10, 1000).reshape(-1, 1)
y_probability = log_reg.predict_proba(x_prediction)
decision_boundary = x_prediction[y_probability[:, 1] >= 0.5][0, 0]

print(f"Decision Boundary (Porosity (%)) = {decision_boundary:.3f}%")
print(
    f"Predicted Class to the left and right (+-0.5) of D.B:"
    f" {log_reg.predict([[decision_boundary - 0.5], [decision_boundary + 0.5]])}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x_train[y_train == 0], y_train[y_train == 0], "bs")
ax.plot(x_train[y_train == 1], y_train[y_train == 1], "r^")
ax.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
ax.plot(x_prediction, y_probability[:, 1], "r-", linewidth=2, label="Sandstone")
ax.plot(x_prediction, y_probability[:, 0], "b--", linewidth=2, label="Not Sandstone")
ax.text(decision_boundary + 0.15, 0.05, "Decision  boundary", fontsize=14, color="k", ha="center", rotation=90)
ax.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
ax.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='r', ec='r')
ax.set_xlabel(feature_name, fontsize=18)
ax.set_ylabel("Probability", fontsize=18)
ax.legend(loc="center left", fontsize=18)
ax.axis([2, 10, -0.02, 1.02])

fig.tight_layout()
plt.show()
