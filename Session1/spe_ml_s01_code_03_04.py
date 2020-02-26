import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

font_dict_title = {'fontsize': 18, 'fontweight': 'bold'}
font_dict_axes = {'fontsize': 14, 'fontweight': 'bold'}
font_legend = {'size': 14, 'weight': 'bold'}

actual_capex = np.array([[46], [48], [50], [54], [58.5]])
actual_npv = np.array([70, 90.2, 130, 175.1, 180])

sl_model = LinearRegression()
sl_model.fit(actual_capex, actual_npv)

# code 3
fig, ax = plt.subplots()

ax.set_title("NPV vs CAPEX", fontdict=font_dict_title)
ax.set_xlabel("CAPEX, MM$", fontdict=font_dict_axes)
ax.set_ylabel("NPV, MM$", fontdict=font_dict_axes)
ax.set_xlim((40, 60))
ax.set_ylim((40, 200))
ax.plot([40, 60], [127, 127], 'g:', lw=3, label='1st ft-line')
ax.plot([45, 57], [40, 200], c='#4b88fa', ls='-', lw=3, label='2nd ft-line')
ax.plot([42, 59], [40, 200], 'k--', lw=2, label='3rd ft-line')
ax.scatter(actual_capex, actual_npv,
           marker='s', s=50, c='#fa574b',
           label='Train Data')
ax.legend(loc='lower right', prop=font_legend)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True)
plt.tight_layout()

# code 4
residual_npv = []
for (capex, npv) in zip(actual_capex, actual_npv):
    residual_npv.append(sl_model.predict([capex])[0] - npv)

fig, ax = plt.subplots()

ax.set_title("NPV vs CAPEX", fontdict=font_dict_title)
ax.set_xlabel("CAPEX, MM$", fontdict=font_dict_axes)
ax.set_ylabel("NPV, MM$", fontdict=font_dict_axes)
ax.set_xlim((40, 60))
ax.set_ylim((40, 200))
ax.scatter(actual_capex, actual_npv,
           marker='s', s=50, c='#fa574b',
           label='Train Data')

ax.plot([40, 60], [sl_model.predict([[40]])[0], sl_model.predict([[60]])[0]], 'g:', lw=3, label='Line of Best fit')
ax.vlines(actual_capex, actual_npv, np.array(residual_npv) + np.array(actual_npv), label='Residual')

ax.legend(loc='lower right', prop=font_legend)
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()


plt.show()
