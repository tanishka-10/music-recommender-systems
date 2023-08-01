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

cols = ['timbre_' + str(i) for i in range(1,n_timbres+1)]
all_songs = []
for s in range(len(data)): # for every song in the dataframe
    song_timbre_vectors = []
    for t in range(n_timesteps): # for each timestep
        # let's now make a list of the timbre components for song s at timestep t
        time_t_timbres = [data[col].iloc[s][t] for col in cols]
        # now let's add it to our song_timbre_vectors_list
        song_timbre_vectors.append(time_t_timbres)
    # now let's add that song's timbre vectors to the all_songs list
    all_songs.append(song_timbre_vectors)

# finally, convert this to a matrix.
X_new = np.array(all_songs)
print(X_new.shape)
y_new = np.array(y)
y_new.shape
#final import
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.losses import BinaryCrossentropy

# Building the LSTM model
##fill in the dimension sizes here
n_timesteps = 120
n_input = 12
n_output = 1

model = Sequential()
#Choose the number of neurons in the LSTM output.
model.add(LSTM(200, input_shape=(n_timesteps,n_input)))
#Choose the number of units in your dense hidden layer
model.add(Dense(120))
#adding more layers
model.add(Dense(10))

model.add(Dense(n_output))
model.summary()
# new train-test split
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, test_size=.2, random_state=1)

model.compile(
    loss=BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(
    X_new_train, y_new_train, validation_data=(X_new_test, y_new_test), batch_size=50, epochs=30
)
