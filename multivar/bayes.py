import pandas as pd
# df=pd.read_csv('../spam-filter-proj/data/original_multivar_dataset.csv')
df=pd.read_csv('../data/SMSSpamCollectionReformatted.csv')
df.head()

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['email'].values.astype('U'))
print(X.toarray())

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X=X.toarray()
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.3, random_state=55)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
y_pred

from sklearn.metrics import accuracy_score 
accuracy_score (y_test,y_pred )

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score 

accuracy_score (y_test,y_pred )
print("Accuracy Score:", accuracy_score(y_test, y_pred))
