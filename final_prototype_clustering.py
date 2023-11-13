import pandas as pd
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
import gradio as gr
from langdetect import detect, LangDetectException
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import os
from fuzzywuzzy import process, fuzz
import numpy as np

# Spotify API credentials
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=os.environ['MY_SECRET_KEY'],
    client_secret=os.environ['MY_SECRET_KEY']))

def is_english_or_popular(title, popularity, popularity_threshold=40):
    try:
        return detect(title) == 'en' or popularity > popularity_threshold
    except LangDetectException:
        return False

tracks_data = pd.read_csv('tracks.csv')
tracks_data = tracks_data[tracks_data['popularity'] > 30]
tracks_data = tracks_data[tracks_data.apply(lambda x: is_english_or_popular(x['name'], x['popularity']), axis=1)]

features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 
            'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'time_signature']

scaler = StandardScaler()
tracks_scaled = scaler.fit_transform(tracks_data[features])

def find_optimal_clusters(data, max_k=13):
    best_score = -1
    best_k = 2
    best_model = None

    sample_data = data[np.random.choice(data.shape[0], 10000, replace=False), :]

    for k in range(4, max_k + 1):  
        model = Birch(n_clusters=k)
        labels = model.fit_predict(sample_data)
        score = silhouette_score(sample_data, labels)

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model

    return best_k, best_model

model_file = 'best_birch_model.joblib'
if not os.path.isfile(model_file):
    optimal_k, optimal_model = find_optimal_clusters(tracks_scaled)
    print(f"Optimal number of clusters: {optimal_k}")  # Print the optimal number of clusters
    joblib.dump(optimal_model, model_file)
else:
    optimal_model = joblib.load(model_file)

tracks_data['cluster_label'] = optimal_model.predict(tracks_scaled)

threshold = 80
def get_close_matches(df, song, artist, threshold):
    df['combined'] = df['name'] + ' ' + df['artists']
    song_matches = process.extract(song, df['combined'], limit=None, scorer=fuzz.token_set_ratio)
    artist_matches = process.extract(artist, df['combined'], limit=None, scorer=fuzz.token_set_ratio)
    close_song_matches = [df['combined'][index] for match, score, index in song_matches if score >= threshold]
    close_artist_matches = [df['combined'][index] for match, score, index in artist_matches if score >= threshold]
    close_matches = set(close_song_matches).intersection(set(close_artist_matches))
    filtered_df = df[df['combined'].isin(close_matches)]
    df.drop('combined', axis=1, inplace=True)
    return filtered_df.drop('combined', axis=1)

def recommend_songs(song, artist, top_n=5):
    filtered_df = get_close_matches(tracks_data, song, artist, threshold)
    if filtered_df.empty:
        return pd.DataFrame(columns=['Name', 'Artist'])

    cluster_label = filtered_df['cluster_label'].iloc[0]
    similar_songs = tracks_data[tracks_data['cluster_label'] == cluster_label].head(top_n)
    result_df = similar_songs[['name', 'artists']]
    result_df.columns = ['Name', 'Artist']
    
    return result_df

iface = gr.Interface(
    fn=recommend_songs,
    inputs=[
        gr.Textbox(label="Enter Song Name"),
        gr.Textbox(label="Enter Artist Name"), 
        gr.Slider(minimum=1, maximum=10, value=5, label="Number of Recommendations", step=1)
    ],
    outputs=gr.DataFrame()
)

iface.launch()
