from sklearn import svm
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20, criterion='entropy', splitter='random'),
                        # algorithm="SAMME.R")

# clf = GaussianNB()
# clf = RandomForestClassifier()

clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

clf.fit(X_train, y_train)
scores = clf.score(X_test, y_test)
print(scores)
