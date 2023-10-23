import numpy as np
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import metrics

import string
import spacy
np.random.seed(42)

from sklearn.linear_model import LogisticRegression
     
data = pd.read_csv(r"toxic_commnets_500.csv")
data.head()

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
print(stop_words)

punctuations = string.punctuation
print(punctuations)
     
# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)



    # print(doc)
    # print(type(doc))

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

sentence = "I am eating a cheesecake"
spacy_tokenizer(sentence)

count_vector = CountVectorizer(tokenizer = spacy_tokenizer)
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

count_vector.fit_transform(["I am eating cheesecake, I like cheesecake","I am playing kabbadi"]).toarray() 

count_vector.get_feature_names_out()

count_vector.vocabulary_

from sklearn.model_selection import train_test_split

X = data['comment_text'] # the features we want to analyze
ylabels = data['toxic'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2,stratify=ylabels)
     
classifier = LogisticRegression()

X_train_vetcors= count_vector.fit_transform(X_train)
X_test_vetcors= count_vector.transform(X_test)
     
X_train_vetcors.shape

X_test_vetcors.shape

X_train_vetcors.toarray()

classifier.fit(X_train_vetcors,y_train)

predicted = classifier.predict(X_test_vetcors)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))
     
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
X_train_vetcors= tfidf_vector.fit_transform(X_train)
X_test_vetcors= tfidf_vector.transform(X_test)
     
classifier = LogisticRegression()
classifier.fit(X_train_vetcors,y_train)
predicted = classifier.predict(X_test_vetcors)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))
     
