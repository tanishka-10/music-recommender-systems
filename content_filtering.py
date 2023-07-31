#@title Run this to download data and prepare our environment! { display-mode: "form" }

# importing packages and dataset
# import gdown
import ast
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from IPython import display
import ipywidgets as widgets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# music_data_url = 'https://drive.google.com/uc?id=1Q4UAv4FPPOlhmFWHkn4Y83dG6M51xnoM'
!wget 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/data_with_most_lyrics.csv'
music_data_path = './data_with_most_lyrics.csv'
# gdown.download(music_data_url, music_data_path, True)

music_data = pd.read_csv(music_data_path)
music_data = music_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])
music_data['lyrics'] = music_data['lyrics'].str.replace('\n', ' ')
available_songs = music_data["track_name"] + ', ' + music_data["artist_name"]
available_songs = available_songs.tolist()


# past_data_url = 'https://drive.google.com/uc?id=1MIkOcP2JY_foloYAR5-Y60YyRVbRhQMs'
!wget 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Music%20Recommendation/spotify_data_urls.csv'
past_data_path = './spotify_data_urls.csv'
# gdown.download(past_data_url, past_data_path , True)

## Load in data
past_data = pd.read_csv(past_data_path)


class Error(Exception):
    """Base class for other exceptions"""
    pass

class NotSeparableError(Error):
    """Raised when the input value is too small"""
    pass

def fix_genres(row):
  row_new = row['genres'].replace(',', '')
  row_new = row_new.replace("'", "")
  row_new = row_new.replace('"', "")
  if len(row_new)!=0 and row_new[0]=='[':
    return row_new[1:-1]
  return row_new

def find_title_from_index(index):
    return music_data["track_name"][index]
def find_artist_from_index(index):
    return music_data["artist_name"][index]
def find_index_from_title(track_name):
    return music_data.index[music_data.track_name == track_name].values[0]

numerical_features = ['danceability', 'track_popularity']
text_features = ['genres']

def combine_features(row):
    '''
    Loop through all of the features and make a string with all of them combined for one row
    '''
    combined_row = ''
    for feature in text_features:
     for feature in text_features:
      combined_row += str(row[feature])
      combined_row += ' '

    return combined_row
#to use your `combine_features function
# gets rid of null values
if 'genres' in text_features:
  music_data['genres'] = music_data['genres'].fillna('')
  music_data['genres'] = music_data.apply(fix_genres, axis=1)
for feature in text_features:
    music_data[feature] = music_data[feature].fillna('')
music_data["combined_features"] = music_data.apply(combine_features,axis=1)
cv = CountVectorizer()
count_matrix = cv.fit_transform(music_data["combined_features"]) # creates a vector out of our combined features
text_vectors = count_matrix.toarray()
numerical = music_data[numerical_features].to_numpy()
# Next line scales numerical values so features like track_popularity and tempo don't outweigh everything else!
numerical = (numerical - numerical.min(axis=0)) / (numerical.max(axis=0) - numerical.min(axis=0))
song_vectors = np.concatenate((text_vectors, numerical), axis=1)
print(song_vectors)
music_data.head()
##similarity score development
def similarity_score(track1, track2, vectors, metric='cosine'):
  if track1==None or track2==None:
    return None

  similarity_matrix = all_similarity(vectors, metric)

  # YOUR CODE HERE (get rid of the score=0 line)
  song1_index = find_index_from_title(track1)
  song2_index = find_index_from_title(track2)
  score = similarity_matrix[song1_index][song2_index]

  return score
# Run this to choose your first song
song_1 = widgets.Combobox(
    placeholder='Choose a Song',
    options=available_songs,
    description='song1:',
    ensure_option=True,
    disabled=False)
song_1
# Run this to choose your second song
song_2 = widgets.Combobox(
    placeholder='Choose a Song',
    options=available_songs,
    description='song2:',
    ensure_option=True,
    disabled=False)
song_2

# Run this to process your song choices!
if song_1.value!='' and song_2.value!='':
  song1 = music_data['track_name'][available_songs.index(song_1.value)]
  song2 = music_data['track_name'][available_songs.index(song_2.value)]

score = similarity_score(song1, song2, song_vectors)
print(score)

##using KNN to create similarity score
song = 'Party In The U.S.A.'
similarity_matrix = all_similarity(song_vectors)
song_index = find_index_from_title(song)
song_similarity = similarity_matrix[song_index]
K = 5
song_indices = []
for i in range(0, K):
  current_max = 0
  current_index = None
  for j in range(len(song_similarity)):
    if song_similarity[j] > current_max and j not in song_indices and j != song_index:
      current_max = song_similarity[j]
      current_index = j
  song_indices.append(current_index)
similar_songs = list(enumerate(song_similarity))
sorted_similar_songs = sorted(similar_songs,key=lambda x:x[1],reverse=True)[1:]
song_indices = [x[0] for x in sorted_similar_songs[:K]]
song_names = []
singers = []
for index in song_indices:
  song_names.append(find_title_from_index(index))
  singers.append(find_artist_from_index(index))

def k_most_similar_songs(song, song_vectors, K):
  similarity_matrix = all_similarity(song_vectors)
  song_index = find_index_from_title(song)
  song_similarity = similarity_matrix[song_index]
  song_indices = []
  for i in range(0, K):
    current_max = 0
    current_index = None
    for j in range(len(song_similarity)):
      if song_similarity[j] > current_max and j not in song_indices and j != song_index:
        current_max = song_similarity[j]
        current_index = j
    song_indices.append(current_index)

  song_names = []
  singers = []
  for index in song_indices:
    song_names.append(find_title_from_index(index))
    singers.append(find_artist_from_index(index))
  return song_names, singers
# to choose a song
song = widgets.Combobox(
    placeholder='Choose a Song',
    options=available_songs,
    description='Song:',
    ensure_option=True,
    disabled=False)
song
#to get your recommended songs
song_name = music_data['track_name'][available_songs.index(song.value)]
similar_songs, similar_artists = k_most_similar_songs(song_name, song_vectors, 10)
print(similar_songs)
print(similar_artists)
