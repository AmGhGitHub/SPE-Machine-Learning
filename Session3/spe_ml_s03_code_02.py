import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)

df_rock = pd.read_csv('data/rock_type.csv', header=[0])
feature_name = ['Porosity (%)', 'Permeability (md)']
x = df_rock[feature_name].values
# convert categorical data to numerical
y = (df_rock['Rock Type'] == 'SS').astype(np.int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

log_reg = LogisticRegression(solver="lbfgs", C=10 ** 20, random_state=42)
log_reg.fit(x_train, y_train)

x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),
    np.linspace(0, 100, 1000).reshape(-1, 1))
x_predicted = np.c_[x0.ravel(), x1.ravel()]

y_probability = log_reg.predict_proba(x_predicted)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_train[y_train == 0, 0], x_train[y_train == 0, 1], "bs")
ax.plot(x_train[y_train == 1, 0], x_train[y_train == 1, 1], "g^")

zz = y_probability[:, 1].reshape(x0.shape)
contour = ax.contour(x0, x1, zz, cmap=plt.get_cmap('brg'))

beta0 = log_reg.intercept_[0]
beta1 = log_reg.coef_[0, 0]
beta2 = log_reg.coef_[0, 1]

left_right = np.array([1, 8])
val_prob_50 = 0.0
# what is the value for other probabilities?
boundary = -(val_prob_50 + beta1 * left_right + beta0) / beta2

ax.clabel(contour, inline=1, fontsize=12)
ax.plot(left_right, boundary, "k--", linewidth=3)
ax.text(7, 90, "Sandstone", fontsize=18, color="g", ha="center")
ax.text(2, 30, "Not Sandstone", fontsize=18, color="b", ha="center")
ax.set_xlabel(feature_name[0], fontsize=18)
ax.set_ylabel(feature_name[1], fontsize=18)
ax.set_title('Binary classification Using Logistic Regressor', fontsize=20)
ax.axis([0, 8, 0, 100])

fig.tight_layout()
plt.show()
