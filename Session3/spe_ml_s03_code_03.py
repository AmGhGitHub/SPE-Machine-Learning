import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)

df_rock = pd.read_csv('data/rock_type_imbalanced.csv')
print("*********** Statistics of the original Dataframe ***********")
print(df_rock.info())
# the values of df_rock.info() indicates that we have na entries
df_rock.dropna(axis=0, inplace=True)  # remove all na values
print("\n\n*********** Statistics of the modified Dataframe ***********")
print(df_rock.info())
# get all unique values in Rock Type (response) column
print("\n\n*********** Statistics of the 'Rock Type' column ***********\n")
print(df_rock['Rock Type'].value_counts())

x = df_rock.iloc[:, :-1]
y = df_rock.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_num = label_encoder.fit_transform(y_train)
y_test_num = label_encoder.transform(y_test)
# print(y_train_num)
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train_num)
y_predict = log_reg.predict(x_test)
cfm = confusion_matrix(label_encoder.inverse_transform(y_test_num), label_encoder.inverse_transform(y_predict))
print(f"Model's Accuracy: {100 * accuracy_score(y_test_num, y_predict):0.1f}%")
print(f"Model's Precision: {100 * precision_score(y_test_num, y_predict):0.1f}%")
print(f"Model's Recall: {100 * recall_score(y_test_num, y_predict):0.1f}%")
print(f"Model's F1 Score: {100 * f1_score(y_test_num, y_predict):0.1f}%")
print('\n********* Classification Report *********')
print(classification_report(label_encoder.inverse_transform(y_test_num), label_encoder.inverse_transform(y_predict)))

fig, ax = plt.subplots(figsize=(13, 10))
ax.matshow(cfm, cmap=plt.get_cmap('rainbow'), alpha=0.5)
for i in range(cfm.shape[0]):
    for j in range(cfm.shape[1]):
        ax.text(x=j, y=i, s=cfm[i, j], va='center', ha='center', fontsize=40)
labels = ['Carbonate', 'Sandstone']
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, minor=False)
ax.set_yticklabels([" "] + labels,
                   minor=False, rotation=90,
                   va='center')
ax.tick_params(axis='both', labelsize=28, length=0)
ax.xaxis.set_label_position('top')
ax.tick_params(axis='both', bottom=False, top=True)
ax.set_xlabel('Predicted', fontsize=46)
ax.set_ylabel('Actual', fontsize=46)

plt.show()
