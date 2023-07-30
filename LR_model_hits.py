#@title Before we begin... import statements! Run this.
# General Imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import gdown

# for the ML
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
#@title Run this to download your data

!wget -O ./spotify_data_urls.csv 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/spotify_data_urls.csv'
data_path = './spotify_data_urls.csv'
# Load in data
data = pd.read_csv(data_path)
basic_data = data[['Artist','Track','Year','url','Label']]
X_2 = [['dancebility', 'instrumentalness', 'speechiness', 'energy', 'acousticness', 'loudness']] # Whatever features you'd like!
y_2 = data[['Label']]

# Step 1: Split our data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Step 2: Initialize our logistic regression model and train it
lr = LogisticRegression()
lr.fit(X_train,y_train)
# Step 3: Calculate and print the score
lr_score = lr.score(X_test,y_test)
print ("LR Score:", lr_score)

## LR model in loop
log_reg_scores =[]
for i in range(10):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  lr = LogisticRegression()
  lr.fit(X_train,y_train)
  log_reg_scores.append(lr.score(X_test,y_test))


print("Average logistic regression score is:", sum(log_reg_scores)/len(log_reg_scores))

## Trying different classifiers other than scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
print('KNN score',knn.score(X_test,y_test))
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('GNB score', gnb.score(X_test, y_test))
mlp = MLPClassifier(hidden_layer_sizes=(20,10,5))
mlp.fit(X_train, y_train)
print('MLP score', mlp.score(X_test, y_test))
svc = SVC()
svc.fit(X_train,y_train)
print('SVC score',svc.score(X_test,y_test))