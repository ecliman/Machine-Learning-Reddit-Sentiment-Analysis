import numpy as np
import scipy.sparse as spload
import pandas as pd
import sklearn.model_selection as validation
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score

y_train = np.load('../data/y_train.npy')

x_train_svd = np.load('../data/xtrainsvd.npy')
x_test_svd = np.load('../data/xtestsvd.npy')

x_train = spload.load_npz('../data/xtrainraw.npz')
x_test = spload.load_npz('../data/xtestraw.npz')

x_trainbin = spload.load_npz('../data/xtrainbin.npz')
x_testbin = spload.load_npz('../data/xtestbin.npz')

x_traind2v = np.load('../data/train_d2v.npy')
x_testd2v = np.load('../data/test_d2v.npy')

all_xtrain = [x_train_svd, x_train, x_traind2v]
all_xtest = [x_test_svd, x_test, x_test_svd]

# Done importing

enc = preprocessing.LabelEncoder()

enc.fit(["nba","hockey","leagueoflegends","soccer","funny","movies","anime","Overwatch","trees","GlobalOffensive","nfl","AskReddit",
         "gameofthrones","conspiracy","worldnews","Music","wow","europe","canada","baseball"])


# Logistic Regression
logreg = LogisticRegression(random_state=0, tol=0.1, solver='saga', l1_ratio=0.8, penalty='elasticnet', n_jobs=4)
logreg.fit(x_train, y_train)
print('Logistic Regression Fitted.')

scores = cross_val_score(logreg, x_train, y_train, cv=3)
print('Log Reg Scores', scores)

# CNB
cnb = ComplementNB()
cnb.fit(x_train, y_train)
print('CB fitted')
scores = cross_val_score(cnb, x_train, y_train, cv=3)
print('CB Scores', scores)

pred = cnb.predict(x_test)
pred = enc.inverse_transform(pred)
pd.DataFrame(pred).to_csv('../results/cnb.csv')

# Nearest Neighbours
# nearest = KNeighborsClassifier(n_neighbors=5, n_jobs=4, leaf_size=100)
# nearest.fit(x_train, y_train)
# print('Nearest Neighbors Fitted.')

# scores = cross_val_score(nearest, x_train, y_train, cv=2)
# print('Nearest Neighbors Scores', scores)

# Random Forests
forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, n_jobs=4)
forest.fit(x_train, y_train)
print('Random Forest Fitted.')

scores = cross_val_score(forest, x_train, y_train, cv=5)
print('Random Forest Scores', scores)

# Ada boost
ada = AdaBoostClassifier()
ada.fit(x_train, y_train)
print('AdaBoost fitted')
scores = cross_val_score(ada, x_train, y_train, cv=5)
print('AdaBoost scores', scores)


# pred = forest.predict(x_test_svd)
# pred = enc.inverse_transform(pred)
# pd.DataFrame(pred).to_csv('../results/forest.csv')






# pred = logreg.predict(x_test)
# pred = enc.inverse_transform(pred)
# pd.DataFrame(pred).to_csv('../results/logreg.csv')








# Decision Tree
# tree = DecisionTreeClassifier(random_state=0, min_samples_leaf=3)
# tree.fit(x_train_svd, y_train)
# print ('Decision Tree Fitted.')
#
#
# scores = cross_val_score(tree, x_train_svd, y_train, cv=5)
# print('Decision Tree Scores', scores)
#
# pred = logreg.predict(x_test_svd)
# pred = enc.inverse_transform(pred)
# pd.DataFrame(pred).to_csv('../results/tree.csv')
#
# # SVM
# svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,random_state=0,max_iter=5, tol=None)
# svm.fit(x_train, y_train)
#
# scores = cross_val_score(svm, x_train_svd, y_train, cv=5)
# print('SVM', scores)
