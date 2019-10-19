X_train,y_train,X_test  are raw data

y is encoded with 1~20 representing different subreddits

I tooke away the URL and @user from the data,
and apply lemmalization with inspiration by 
https://simonhessner.de/lemmatize-whole-sentences-with-python-and-nltks-wordnetlemmatizer/?source=post_page-----c1bfff963258----------------------



Xtrain,Xtest are matrix using TfidfVectorizer
with shape
(70000,220000)
(30000,220000+)  where I use stopwords from ntlk and ngrams 
                    ranged (1,2)



Then I also did dim reduction by using featureselection
X_train_new  (70000,100000)
X_test_new     (30000,10000)

add XtrainBin  with 10000 max features
    XtestBin as binary sparse matrix
