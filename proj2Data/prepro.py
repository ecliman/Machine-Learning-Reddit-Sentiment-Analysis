import pandas as pd
import numpy
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2 
from sklearn.feature_extraction.text import CountVectorizer


Data = pd.read_csv("reddit_train.csv",sep=",",usecols=[1,2])
Data = Data.sample(frac=1).reset_index(drop=True)

Test = pd.read_csv("reddit_test.csv",sep=",")
Test = Test.sample(frac=1).reset_index(drop=True)





stopWords = set(stopwords.words('english'))
stopWords.add('URL')
stopWords.add('AT_USER')






lemmatizer = WordNetLemmatizer()


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:        
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
#This lemmatization part is inspired by 'Lemmatize whole sentences with Python and nltk’s WordNetLemmatizer' see reference.


enc = preprocessing.LabelEncoder()

enc.fit(["nba","hockey","leagueoflegends","soccer","funny","movies","anime","Overwatch","trees","GlobalOffensive","nfl","AskReddit",
         "gameofthrones","conspiracy","worldnews","Music","wow","europe","canada","baseball"])


Data['comments']=Data['comments'].replace(to_replace=r'((www\.[^\s]+)|(https?://[^\s]+))', value='URL', regex=True)
Data['comments']=Data['comments'].replace(to_replace=r'@[^\s]+', value='AT_USER', regex=True)
Data['comments']=Data.apply(lambda row: lemmatize_sentence(row['comments']), axis=1)


X_train= Data['comments']
y_train= enc.transform(Data['subreddits'])
X_test= Test['comments']










vec = TfidfVectorizer(stop_words=stopWords, ngram_range=(1, 2))
Xtrain=vec.fit_transform(X_train)
Xtest= vec.transform(X_test)



selection = SelectKBest(chi2, k=100000)
X_train_new = selection.fit_transform(Xtrain,y_train)
X_test_new = selection.transform(Xtest)



binaryVec = CountVectorizer(stop_words=stopWords,ngram_range=(1,2),binary=True,max_features=10000)
XtrainBin=binaryVec.fit_transform(X_train)
XtestBin=binaryVec.transform(X_test)



from sklearn.naive_bayes import ComplementNB
cnb=ComplementNB()
from sklearn.ensemble import BaggingClassifier

bootstrap = BaggingClassifier (n_estimators=200,base_estimator=cnb)
bootstrap.fit(X_train_new,y_train)
pred111=bootstrap.predict(X_test_new)

pred = enc.inverse_transform(pred111)
pd.DataFrame(pred, columns=['Id','Category']).to_csv('pred.csv')
