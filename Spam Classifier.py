# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 10:33:13 2020

@author: jainrohi

98.38 % accuracy with Naive bayes
"""

import pandas as pd


messages = pd.read_csv('C:/Users/jainrohi/Desktop/NLP/Spam Classifier/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

processed = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    processed.append(review)

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(processed).toarray()


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 101)


# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)

y_pred= model.predict(X_test)


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
 