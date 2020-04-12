# Machine-Learning-Reddit-Sentiment-Analysis

The object of this project was to classify Reddit comments input words into their respective
subreddits. In order to do this, our task was to extract features from the comments and implement various
classification models from sci-kit learn such as a Support Vector Machine (SVM with the SGDClassifier),
Logistic Regression, Ridge Classifier, Complement Naive Bayes and a Gaussian Naive Bayes. We also
evaluated the effect of employing sci-kit learn’s ensemble methods, namely VotingClassifier (stacking)
and BaggingClassifier. Finally, we implemented a Bernoulli Naive Bayes which uses binary occurrence
matrix instead of real values. The dataset that these models were tested on has 30000 datapoints, each fell
into one of 20 possible subreddits. Our testing shows Complement Naïve Bayes as the best base model,
and a stacking ensemble of all the models slightly beat our best base model
