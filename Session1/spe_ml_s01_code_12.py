# code 12
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train = np.array([[6], [8], [10], [14], [18]])
y_train = np.array([7, 9, 13, 17.5, 18])
x_test = np.array([[6], [8], [11], [16]])
y_test = np.array([8, 12, 15, 18])

r2_n = []

for idx, n in enumerate(range(1, 11)):
    poly_featurizer = PolynomialFeatures(degree=n)
    x_train_poly = poly_featurizer.fit_transform(x_train)
    pl_regressor = LinearRegression()
    pl_regressor.fit(x_train_poly, y_train)
    x_test_poly = poly_featurizer.transform(x_test)
    _r2 = pl_regressor.score(x_test_poly, y_test)
    r2_n.append(_r2)
    print(f"n={n} --> R2= {100 * _r2:.1f}%")

font_dict = {'fontsize': 20, 'fontweight': 'bold'}

fig, ax = plt.subplots()
ax.plot(range(1, 11), r2_n, 'g', lw=3, marker='s', markersize=7, markerfacecolor='white')
ax.set_ylim([-0.2, 1.0])
_xticks = list(range(1, 11, 1))
ax.set_xticks(_xticks)
ax.set_xticklabels(_xticks, fontdict={'fontsize': 16})
ax.set_xlabel('n', **font_dict)
ax.set_ylabel('R^2', **font_dict)
ax.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.show()
