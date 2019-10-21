from sklearn.naive_bayes import ComplementNB,MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier,BaggingClassifier


lr = LogisticRegression(random_state=0,multi_class='ovr',solver='liblinear',penalty='l2')
cnb=ComplementNB()
mnb=MultinomialNB()
rc=RidgeClassifier()
sgd=SGDClassifier()


voting = VotingClassifier(estimators=[('cnb',cnb),('mnb',mnb),('lr',lr),('sgd',sgd),('rc',rc)],voting='hard')
bootstrap = BaggingClassifier(base_estimator=cnb,n_estimator=100)


