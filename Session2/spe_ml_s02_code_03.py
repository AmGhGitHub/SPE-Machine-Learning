import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

df = pd.read_csv('perm_vs_sw_data.csv', index_col=[0])
x = df.index.values.reshape(-1, 1)
y = df[df.columns[0]].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.1, random_state=42)

fig, ax = plt.subplots()
for n, style, width in zip((1, 2, 17), ('r-+', 'b--', 'g-'), (2, 3, 2)):
    poly_features = PolynomialFeatures(degree=n, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly, y_train)
    X_test_poly = poly_features.transform(X_test)
    model_r2 = lin_reg.score(X_test_poly, y_test)
    print(model_r2)
    X_new = np.linspace(0, 5, 200).reshape(-1, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)
    ax.plot(X_new, y_new, style, linewidth=width, label=f"n= {n}")

ax.plot(x[::2], y[::2], '.', markersize=10, markeredgecolor='black', markerfacecolor='white')
ax.set_title('Permeability vs. Porosity', fontdict={'fontsize': 18, 'weight': 'bold'})
ax.set_xlabel(r'$\phi$, %', fontsize=16)
ax.set_ylabel(r'$k, md$', fontsize=16)
ax.set(xlim=[0, 5], ylim=[0, 70])
ax.legend(loc='upper left', fontsize=18)

fig.tight_layout()
plt.show()
