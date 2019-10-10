X_train,y_train,X_test  are raw data

y is encoded with 1~20 representing different subreddits

I tooke away the URL and @user from the data,
and apply tokenizer from ntlk



Xtrain,Xtest are matrix using TfidfVectorizer
with shape
(70000,2million+)
(30000,2million+)  where I use stopwords from ntlk and ngrams 
                    ranged (1,3)



Then I also did SVD on the sets.
XtrainSVD  (70000,100)
XtestSV     (30000,100)
