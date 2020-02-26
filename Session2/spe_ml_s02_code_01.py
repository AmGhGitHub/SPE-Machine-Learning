import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)


def format_data(x, add_column_one=False):
    x = x.copy()  # make a copy of input vector
    if add_column_one:
        return np.c_[np.ones((x.shape[0], 1)), x.reshape(-1, 1)]
    else:
        return x.reshape(-1, 1)


def plot_gradient_descent(ax, x, y, eta, max_iter=500, random_state=42):
    ax.plot(x, y, "r.")

    np.random.seed(random_state)
    beta = np.random.randn(2, 1)  # random initialization of the coefficients

    X = format_data(x, True)
    y = format_data(y, False)

    x_new = np.array([6, 9])
    X_new = format_data(x_new, True)

    m = len(x)
    for itr in range(max_iter):
        if itr < 20:
            y_predict = X_new.dot(beta)
            ax.plot(x_new, y_predict, ls='-', c='#00ffcc')

        delta_beta = (2.0 / m) * X.T.dot(X.dot(beta) - y)
        beta -= eta * delta_beta
    ax.set_xlabel(r"$S_w , \%$", fontsize=22)
    ax.set_title(fr"$\eta = {eta}$", fontsize=18)
    ax.set_xlim([6, 9])
    ax.set_ylim([11, 23])


df = pd.read_csv('poro_vs_sw.csv')
x = df['porosity-%'].values
y = df['sw-%'].values

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_ylabel(r"$\phi, \%$", fontsize=22)
plot_gradient_descent(ax1, x, y, eta=0.0008)
plot_gradient_descent(ax2, x, y, eta=0.0016)
plot_gradient_descent(ax3, x, y, eta=0.016, random_state=52)

plt.tight_layout()
plt.show()
