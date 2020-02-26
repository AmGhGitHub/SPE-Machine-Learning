import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)


def plot_model(ax, model_class, polynomial, alphas, **kwargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **kwargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                ("std_scaler", StandardScaler()),
                ("regul_reg", model),
            ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        ax.plot(X_new, y_new_regul, style, linewidth=lw, label=fr"$\alpha = {alpha}$")
    ax.plot(X, y, "b.", linewidth=3)
    ax.legend(loc="upper left", fontsize=16)
    ax.set_xlabel("$x$", fontsize=18)
    ax.set(xlim=(0, 3), ylim=(0, 4))


fig, ax = plt.subplots(1, 2, **{'figsize': (12, 6)})
ax[0].set_ylabel("$y$", rotation=0, fontsize=18)

plot_model(ax[0], Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plot_model(ax[1], Ridge, polynomial=True, alphas=(0, 10 ** -5, 1), random_state=42)

fig.tight_layout()
plt.show()
