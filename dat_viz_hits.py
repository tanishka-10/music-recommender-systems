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
# Danceability Histogram
plot1 = plt.figure(1)
plt.hist(data["danceability"], color = 'b')
plt.xlabel('Danceability')
plt.ylabel('Count')
plt.title('Danceability Histogram')
plt.plot()

# Energy Histogram code below!
plot2 = plt.figure(2)
plt.hist(data["energy"], color = 'b')
plt.xlabel('Energy')
plt.ylabel('Count')
plt.title('Energy Histogram')
plt.plot()
# Key Histogram code below!
plot3 = plt.figure(3)
plt.hist(data["key"], color = 'b')
plt.xlabel('Key')
plt.ylabel('Count')
plt.title('Key Histogram')
plt.plot()
# Loudness Histogram code below!
plot4 = plt.figure(4)
plt.hist(data["loudness"], color = 'b')
plt.xlabel('Loudness')
plt.ylabel('Count')
plt.title('Loudness Histogram')
plt.plot()
plt.show()

#subjective features list below
subjective = data[['index', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']] # FILL IN SUBJECTIVE FEATURES
subjective.head(10) #first 10

#objective features
objective = data[['Artist', 'Track', 'Year', 'key', 'loudness', 'mode', 'tempo']] # FILL IN OBJECTIVE FEATURES
objective.head(10) #first 10

# hit = 1 and non hit = 0
outputs = data[['Artist','Track','Label']]
outputs.head(10)
