import pandas as pd

import numpy
from sklearn import preprocessing
from nltk.corpus import stopwords

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


Data = pd.read_csv("reddit_train.csv",sep=",",usecols=[1,2])
Data = Data.sample(frac=1).reset_index(drop=True)

Test = pd.read_csv("reddit_test.csv",sep=",")
Test = Test.sample(frac=1).reset_index(drop=True)





stopWords = set(stopwords.words('english'))
stopWords.add('URL')
stopWords.add('AT_USER')




enc = preprocessing.LabelEncoder()

enc.fit(["nba","hockey","leagueoflegends","soccer","funny","movies","anime","Overwatch","trees","GlobalOffensive","nfl","AskReddit",
         "gameofthrones","conspiracy","worldnews","Music","wow","europe","canada","baseball"])


Data['comments']=Data['comments'].replace(to_replace=r'((www\.[^\s]+)|(https?://[^\s]+))', value='URL', regex=True)
Data['comments']=Data['comments'].replace(to_replace=r'@[^\s]+', value='AT_USER', regex=True)
Data['comments'].apply(word_tokenize)


X_train= Data['comments']
y_train= enc.transform(Data['subreddits'])
X_test= Test['comments']










vec = TfidfVectorizer(stop_words=stopWords, ngram_range=(1, 3))
 

Xtrain=vec.fit_transform(X_train)
Xtest= vec.transform(X_test)






svd = TruncatedSVD(n_components=100)
XtrainSVD=svd.fit_transform(Xtrain)
XtestSVD=svd.transform(Xtest)