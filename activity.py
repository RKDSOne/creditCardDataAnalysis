import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('creditcard.csv')
feature_cols = ['Time', 'V1', 'V2']
X = df.loc[:, feature_cols]
y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.43)
sm = SMOTE(kind='regular')
X_res, y_res = sm.fit_sample(X_train, y_train)
print('Saving training data and it\'s labels')
np.save('X_train.npy', X_res)
np.save('y_train.npy', y_res)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

