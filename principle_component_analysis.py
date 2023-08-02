# to download data and prepare the environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

# audio analysis
import librosa
import librosa.display
!pip install audio2numpy
!apt-get install ffmpeg
import audio2numpy
import IPython.display as ipd
from IPython.display import YouTubeVideo

#warning supression
import warnings
warnings.filterwarnings("ignore")

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

# chirp file
# chirp_url = 'https://drive.google.com/uc?id=1iX6wV0cSGIVM0nTItTSUlUehhKCILymB'
!wget -O ./chirp.wav = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/chirp.wav'
chirp_path = './chirp.wav'

# two wav files
# wav1_url = 'https://drive.google.com/uc?id=14qA48pPVJKU4KlP8YwqTpDo7WEewjXwf'
!wget -O ./sample1.wav = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/track_3504.wav'
wav1_path = './sample1.wav'

# wav2_url = 'https://drive.google.com/uc?id=12XIWhMCAiabzGqXQhN2wB9o5p6MFR4Y0'
!wget -O ./sample2.wav = 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/track_3642.wav'
wav2_path = './sample2.wav'

# Check that the data downloaded by running this cell! It'll have more columns than usual.
data = pd.read_csv(data_path)
data.rename(columns={'Unnamed: 0':'orig_index'},inplace=True)
data.drop(columns=['index'],inplace=True)

# generates some slightly noisy linearly related data and plots it
rng = np.random.RandomState(0)
n_samples = 500
cov = [[3, 3],
       [2, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)
plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1])
plt.gca().set(aspect='equal',xlabel='x1',ylabel='x2')
plt.show()

# run PCA
pca = PCA().fit(X)
pca

#Let's now see what the principal components are capturing
plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1])
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)): # this line of code means: get the actual PC vector and the amount of variance it explains
    comp = comp * var  # scale component by its variance explanation power
    print((comp[0]**2 + comp[1]**2)**(0.5))
    # then plot it as a line on top of the data
    plt.plot([0, comp[0]], [0, comp[1]], label=f"component {i}", linewidth=5,
             color=f"C{i + 2}")
plt.gca().set(xlabel='x1', ylabel='x2',aspect='equal')
plt.legend()
plt.show()

print(f'Number of frequency bins: {specgram_db.shape[0]}')


pca_specgram = PCA(n_components=.9).fit(specgram_db.T)
print(f'number of components needed for 90% variance: {pca_specgram.n_components_}')

# pc plotting function
def plot_pc(pc_idx):
    pc_fig, pc_ax = plt.subplots()
    pc_ax = sns.heatmap([pca_specgram.components_[pc_idx]])
    pc_fig.set_size_inches(10,2)
    pc_ax.set_xlabel('frequency bin index')
    pc_ax.set_yticklabels('')
    pc_ax.set_title(f'PC{pc_idx+1}')
    plt.show()

for i in range(5):
  plot_pc(i)
#Projects the PCs back onto the original spectrogram to see what effect that has on the visualization
mu = np.mean(specgram_db.T, axis=0)
specgram_db_pc = np.matmul(pca_specgram.transform(specgram_db.T), pca_specgram.components_)
specgram_db_pc = specgram_db_pc + mu
specgram_db_pc = specgram_db_pc.T

fig, ax = plt.subplots()
img = librosa.display.specshow(specgram_db_pc, ax=ax, x_axis='time', y_axis='log')
fig.colorbar(img, ax=ax)
fig.set_size_inches(20,10)