# Run this to download data and prepare the environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# import gdown
import sys

# new dependencies
!pip --disable-pip-version-check install requests==2.27.1
!pip --disable-pip-version-check install urllib3
!pip --disable-pip-version-check install spotipy --upgrade
!pip --disable-pip-version-check  install traces

import spotipy
import requests
from spotipy.oauth2 import SpotifyClientCredentials
import traces

# do not change these codes
client_credentials_manager = SpotifyClientCredentials('e316c18604cb42399f3b679791362112','4bd95a0d37d145998cfdcaf2a68579d7')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# machine learning models
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#warning supression
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# audio analysis
import librosa
import librosa.display
!pip install audio2numpy
!apt-get install ffmpeg
import audio2numpy
import IPython.display as ipd

# helper function to plot a timbre timecourse
def plot_timbre(df, song_id, timbre_col):
    col = f'timbre_{timbre_col}'
    plt.figure()
    plt.plot([i*.5 for i in range(len(df[col][song_id]))], df[col][song_id])
    plt.xlabel('time/seconds')
    plt.ylabel('timbre component weight')

# timbre timecourses
# data_url = 'https://drive.google.com/uc?id=1jwG1B98Uq5phurfGmdZg8w2BCJdth6Io'
!wget -O ./spotify_data_timbre.csv = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/spotify_data_timbre.csv'
data_path = './spotify_data_timbre.csv'
# gdown.download(data_url, data_path , True)

# chirp file
# chirp_url = 'https://drive.google.com/uc?id=1iX6wV0cSGIVM0nTItTSUlUehhKCILymB'
!wget -O ./chirp.csv = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/chirp.wav'
chirp_path = './chirp.wav'
# gdown.download(chirp_url, chirp_path , True)

# two wav files
# wav1_url = 'https://drive.google.com/uc?id=14qA48pPVJKU4KlP8YwqTpDo7WEewjXwf'
!wget -O ./wav1_path.csv = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/track_3504.wav'
wav1_path = './sample1.wav'
# gdown.download(wav1_url, wav1_path , True)

# wav2_url = 'https://drive.google.com/uc?id=12XIWhMCAiabzGqXQhN2wB9o5p6MFR4Y0'
!wget -O ./wav2_path.csv = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/track_3642.wav'
wav2_path = './sample2.wav'
# gdown.download(wav2_url, wav2_path , True)

# Check that the data downloaded by running this cell! It'll have more columns than usual.
data = pd.read_csv(data_path)
data.rename(columns={'Unnamed: 0':'orig_index'},inplace=True)
data.drop(columns=['index'],inplace=True)


def string_to_list(input_string):
    output_list = []

    temp = input_string[1:-1].split(',')
    for number in temp:
        output_list.append(float(number))

    return output_list


# some tests to see if code worked
test_input = '[5, 12.3, 3.3, 9.7]'
print(string_to_list(test_input))
print(type(string_to_list(test_input)))
for i in range(1,13): # note we need to go from 1 to 13 since range leaves out the last number
    column = 'timbre_' + str(i)
    if type(data[column][0]) == str: # this means that if we rerun the cell accidentally it won't break!
        data[column] = data[column].apply(string_to_list)

data.head()
n_timbres = 12 # this is the number of timbre components
n_timesteps = 120 # this is the number of timesteps we're interested in

for i in range(1,n_timbres+1):
    exploded_cols = []
    for t in range(n_timesteps):
        new_column_title = 'timbre_' + str(i) + '_' + str(t) # this gets us the column title we need
        exploded_cols.append(new_column_title)
    # now, let's generate the new columns
    data[exploded_cols] = pd.DataFrame(data['timbre_' + str(i)].to_list(), index=data.index)
column = "timbre_1_119"
null_values = np.where(data[column].isnull())[0]
data.drop(index= null_values ,inplace=True)

timbre_cols = [f'timbre_{i}_{j}' for i in range(1,n_timbres+1) for j in range(n_timesteps)]


X = data[timbre_cols]
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

models = [LogisticRegression(),
          SVC(),
          KNeighborsClassifier(),
          GaussianNB(),
          MLPClassifier(hidden_layer_sizes=(100,10))]
model_names = [type(model).__name__ for model in models]

model_scores = []

for model in models:
    model.fit(X_train,y_train)
    print(f'{type(model).__name__} test accuracy = {model.score(X_test,y_test)}')
    model_scores.append(model.score(X_test,y_test))

plt.figure(figsize=(10,10))
plt.bar(model_names, model_scores)
plt.show()


