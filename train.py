import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('new_data.csv', index_col=0)
data = data.replace([np.inf, -np.inf], np.nan).dropna()

columns = [column for column in data.columns if column != 'label']
X, y = data[columns], data['label']
# X = StandardScaler().fit(X).transform(X) # transform but get more lower score
X.drop(['hamming_tag'],axis = 1, inplace=True)
X.drop(['distance_cosine_tag_count'],axis = 1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

print('split data done, start training')

rf = RandomForestClassifier(n_estimators=150, n_jobs=8)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
score = rf.score(X_test, y_test)
print('test score of Random Forest is  {}'.format(score))

rf = DecisionTreeClassifier(max_depth=5)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
score = rf.score(X_test, y_test)
print('test score of Decision Tree is {}'.format(score))

rf = svm.SVC(kernel = 'linear')
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
score = rf.score(X_test, y_test)
print('test score of SVM is {}'.format(score))





