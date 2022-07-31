import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import xgboost as xgb 

df = pd.read_csv('cc_data.csv')

df3 = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

dummies = pd.get_dummies(df3['type']).drop(['CASH_IN'], axis=1)

df4 = pd.concat([df3, dummies], axis=1).drop(['type'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df4.drop(['isFraud'], axis=1), df4.isFraud, test_size=0.2, random_state=False)

model = LogisticRegression()

model.fit(X_train, y_train)
model.score(X_test, y_test)

predict = model.predict(X_test)

cm = confusion_matrix(y_test, predict)

sns.heatmap(cm, cmap = 'winter', annot =True, fmt='d',cbar=False, linecolor='Black', linewidths=3)
plt.xticks(np.arange(2)+.5,['No Fraud', 'Fraud'])
plt.yticks(np.arange(2)+.5,['No Fraud', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


