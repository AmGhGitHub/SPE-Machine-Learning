# code 10
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train = np.array([[6], [8], [10], [14], [18]])
y_train = np.array([7, 9, 13, 17.5, 18])
x_test = np.array([[6], [8], [11], [16]])
y_test = np.array([8, 12, 15, 18])

xx = np.arange(0, 25, 0.05)
fig, ax = plt.subplots()
ax.scatter(x_train, y_train, s=60, c='red')
line_style = ["-", "--"]
line_color = ['black', 'blue']
r2_n = []

for idx, n in enumerate([1, 2]):
    poly_featurizer = PolynomialFeatures(degree=n)
    x_train_poly = poly_featurizer.fit_transform(x_train)
    pl_regressor = LinearRegression()
    pl_regressor.fit(x_train_poly, y_train)
    xx_poly = poly_featurizer.transform(xx.reshape(xx.shape[0], 1))
    x_test_poly = poly_featurizer.transform(x_test)
    ax.plot(xx, pl_regressor.predict(xx_poly),
            c=line_color[idx], ls=line_style[idx], label=f"n= {n}")
    _r2 = pl_regressor.score(x_test_poly, y_test)
    r2_n.append(_r2)
    print(f"n={n} --> R2= {100 * pl_regressor.score(x_test_poly, y_test):.1f}%")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim([0, 25])
ax.set_ylim([0, 25])
ax.legend()
plt.tight_layout()
plt.show()
