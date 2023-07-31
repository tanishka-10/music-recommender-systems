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

def parse_string_into_list(string):
  return string[1:len(string)-1].split(', ')

##to use your string parser in the data frame!
def separate_values(column):
  mdata = music_data.copy()
  mdata[column] = mdata.apply(lambda row: parse_string_into_list(row[column]), axis=1)
  mdata.head()

  mdata = mdata.explode(column)
  return mdata

# to choose your genre
available_genres = separate_values('genres')['genres'].value_counts().index.tolist()
genre = widgets.Combobox(
    placeholder='Choose a Genre',
    options=available_genres,
    description='genre:',
    ensure_option=True,
    disabled=False)
genre
if genre.value!=None: # creating the data frame with only tracks from chosen genre
  genre_tracks = separated_genres.loc[separated_genres['genres'] == genre.value]

# setting up logistic regression
X = past_data[['danceability', 'instrumentalness', 'speechiness', 'energy', 'acousticness', 'loudness']]
y = past_data[['Label']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr = LogisticRegression()
lr.fit(X_train,y_train)

 ##Create new X values with music_data
X_new = music_data[['danceability', 'instrumentalness', 'speechiness', 'energy', 'acousticness', 'loudness']]

 ##predict if each song will be hit/not hit and add prediction to music_data
music_data['hit_prediction'] = lr.predict(X_new)

music_data.head() ## 1 = hit and 0 = non hit
##to choose genre
available_genres = separate_values('genres')['genres'].value_counts().index.tolist()
genre = widgets.Combobox(
    placeholder='Choose a Genre',
    options=available_genres,
    description='genre:',
    ensure_option=True,
    disabled=False)
genre

#to re-separate the genres since the data changed
separated_genres = separate_values('genres')

# segmenting data based on chosen genre
genre_tracks = None
if genre.value!=None:
  genre_tracks = separated_genres.loc[separated_genres['genres'] == genre.value]

# segment data based on hits
if genre.value!=None:
  genre_hits = genre_tracks.loc[genre_tracks['hit_prediction'] == 1]
##prints top hits in chosen genre
pd.set_option("display.max_rows", None, "display.max_columns", None)
genre_hits[['track_name', 'artist_name']].head(10)
