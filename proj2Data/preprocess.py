import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Import data
training_set = pd.read_csv('reddit_train.csv', sep=',', usecols=[1,2])
training_set = training_set.sample(frac=1).reset_index(drop=True)

test_set = pd.read_csv('reddit_test.csv', sep=',')
test_set = test_set.sample(frac=1).reset_index(drop=True)

my_stopwords = set(stopwords.words('english'))

# Encode classes
encoder = preprocessing.LabelEncoder()
encoder.fit(["nba","hockey","leagueoflegends","soccer","funny","movies","anime","Overwatch","trees","GlobalOffensive","nfl","AskReddit",
         "gameofthrones","conspiracy","worldnews","Music","wow","europe","canada","baseball"])


# Split x, y
x_train = training_set['comments']
y_train = encoder.transform(training_set['subreddits'])
x_test = test_set['comments']


# Lemmatize x
def lemmatize(data):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for i in range(0, len(data)):
        data[i] = nltk.word_tokenize(data[i])
        data[i] = ' '.join([lemmatizer.lemmatize(w) for w in data[i]])

    return data[i]


# Generate binary occurrence matrix
cv = CountVectorizer(stop_words='english', binary=True)
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

pca = TruncatedSVD(n_components=100)
x_train = pca.fit_transform(x_train)

print(x_train)


