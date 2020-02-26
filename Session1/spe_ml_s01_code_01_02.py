# code 1
import matplotlib.pyplot as plt
import numpy as np

actual_capex = np.array([[46], [48], [50], [54], [58.5]])
actual_npv = np.array([70, 90.2, 130, 175.1, 180])
fig, ax = plt.subplots()
ax.scatter(actual_capex, actual_npv,
           marker='s', s=50, c='#fa574b',
           label='Train Data')
font_dict = {'fontsize': 18, 'fontweight': 'bold'}
ax.set_title("NPV vs CAPEX", **font_dict)
ax.set_xlabel("CAPEX, MM$")
ax.set_ylabel("NPV, MM$")
ax.legend(loc='lower right')
ax.set_xlim((40, 60))
ax.set_ylim((40, 200))
ax.grid(True)
plt.show()

# code 2
from sklearn.linear_model import LinearRegression
sl_model = LinearRegression()
sl_model.fit(actual_capex, actual_npv)
predicted_npv = sl_model.predict(np.array([[52]]))
print(f"CAPEX of $MM 52 should bring about an NPV of $MM {predicted_npv[0]:.1f}")

