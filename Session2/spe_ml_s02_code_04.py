import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# np.random.seed(1)

mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

df = pd.read_csv('perm_vs_sw_data.csv', index_col=[0])
x = df.index.values.reshape(-1, 1)
y = df[df.columns[0]].values.reshape(-1, 1)


def plot_learning_curves(model, x, y):
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.1, random_state=42)
    train_errors, validation_errors = np.zeros(len(x_train)), np.zeros(len(x_train))
    fig, ax = plt.subplots()
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_validation)
        train_errors[m - 1] = mean_squared_error(y_train_predict, y_train[:m])
        validation_errors[m - 1] = mean_squared_error(y_val_predict, y_validation)

    ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    ax.plot(np.sqrt(validation_errors), "b-", linewidth=3, label="val")
    ax.set_xlabel('Training Set Size', fontsize=18)
    ax.set_ylabel('RMSE', fontsize=18)
    ax.set(xlim=(1, 120), ylim=(0, 10))
    fig.tight_layout()


lin_reg = LinearRegression()
polynomial_regression = Pipeline(steps=[
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear_reg", LinearRegression())])
plot_learning_curves(polynomial_regression, x, y)

plt.show()
