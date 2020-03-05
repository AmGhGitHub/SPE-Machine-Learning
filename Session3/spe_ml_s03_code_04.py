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
rock_type_number = {'LM': 0, 'DL': 1, 'SS': 2}

y = df_rock['Rock Type'].map(rock_type_number)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=2, random_state=42, max_iter=10000)
softmax_reg.fit(x_train, y_train)

for idx in range(x_test.shape[0]):
    print(f"Actual y_test: {y_test.values[idx]} --> Predicted y_test: {softmax_reg.predict([x_test[idx]])[0]}")

print(f"Softmax model score =  {softmax_reg.score(x_test, y_test):.2f}")

x0, x1 = np.meshgrid(
    np.linspace(0, 8, 1000).reshape(-1, 1),
    np.linspace(0, 100, 1000).reshape(-1, 1))
x_prediction = np.c_[x0.ravel(), x1.ravel()]

y_probability = softmax_reg.predict_proba(x_prediction)
y_predict = softmax_reg.predict(x_prediction)

z_probability = y_probability[:, 1].reshape(x0.shape)
z_predicted = y_predict.reshape(x0.shape)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_train[y_train == 2, 0], x_train[y_train == 2, 1], "g^", label="Sandstone")
ax.plot(x_train[y_train == 1, 0], x_train[y_train == 1, 1], "bs", label="Dolomite")
ax.plot(x_train[y_train == 0, 0], x_train[y_train == 0, 1], "yo", label="Limestone")

from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

ax.contourf(x0, x1, z_predicted, cmap=custom_cmap)
contour = ax.contour(x0, x1, z_probability, cmap=plt.cm.brg)
ax.clabel(contour, inline=1, fontsize=14)
ax.set_xlabel(feature_name[0], fontsize=18)
ax.set_ylabel(feature_name[1], fontsize=18)
ax.set_title('Mutlti-class classification with Softmax classifier', fontsize=20)
ax.legend(loc="center left", fontsize=18)
plt.show()
