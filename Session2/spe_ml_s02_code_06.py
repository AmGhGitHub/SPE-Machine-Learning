import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
X_new = np.linspace(0, 3, 100).reshape(100, 1)
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5

elastic_net = ElasticNet(alpha=0.001, l1_ratio=0.5)
elastic_net.fit(X, y)
prediction = elastic_net.predict([[1.5]])
print(prediction[0])

def plot_model(ax, model_class, alphas, l1_ratios, **kwargs):
    for ax, l1_ratio in zip(ax, l1_ratios):
        for alpha in alphas:
            model = model_class(alpha, l1_ratio, **kwargs)
            model = Pipeline([
                ("poly_features", PolynomialFeatures(degree=6, include_bias=False)),
                ("std_scaler", StandardScaler()),
                ("regul_reg", model),
            ])
            model.fit(X, y)
            y_new_regul = model.predict(X_new)
            lw = 2 if alpha > 0 else 1
            ax.plot(X_new, y_new_regul, linewidth=lw, label=fr"$\alpha = {alpha}$")
        ax.plot(X, y, "b.", linewidth=3)
        ax.legend(loc="upper left", fontsize=16)
        ax.set_xlabel("$x$", fontsize=18)
        ax.set_title(f"r = {l1_ratio}", fontsize=22)
        ax.set(xlim=(0, 3), ylim=(0, 4))


fig, ax = plt.subplots(1, 2, **{'figsize': (10, 6)})
ax[0].set_ylabel("$y$", rotation=0, fontsize=18)
alphas = 10 ** np.arange(-8, 1, 2, dtype=float)
plot_model(ax, ElasticNet, alphas, l1_ratios=(0.1, 0.9), random_state=42, max_iter=50000)

fig.tight_layout()
# plt.show()
