X_train,y_train,X_test  are raw data

y is encoded with 1~20 representing different subreddits

I tooke away the URL and @user from the data,
and apply lemmalization with inspiration by 
https://simonhessner.de/lemmatize-whole-sentences-with-python-and-nltks-wordnetlemmatizer/?source=post_page-----c1bfff963258----------------------



Xtrain,Xtest are matrix using TfidfVectorizer
with shape
(70000,60000+)
(30000,60000+)  where I use stopwords from ntlk and ngrams 
                    ranged (1,1)



Then I also did SVD on the sets.
XtrainSVD  (70000,5000)
XtestSV     (30000,5000)

add XtrainBin  with 10000 max features
    XtestBin as binary sparse matrix
