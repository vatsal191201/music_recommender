import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.sparse import csr_matrix
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from fuzzywuzzy import process, fuzz
import gradio as gr
import time
from fuzzywuzzy import fuzz
import re


# Spotify API setup
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id='76ff1108754147b0b59b9beb48e6af37',
    client_secret='32a408639a054cfead2b0432ac5b1f59'
))

# Define features at the top of your script
features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

def is_english_title(title):
    # Extended regular expression to include Unicode letters and numbers
    pattern = re.compile(r"^[A-Za-z0-9 \-\!\”\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\\^\_\`\{\|\}\~]+$")
    return pattern.fullmatch(title) is not None

def read_data(path, scaler_option=0):
    features = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    try:
        tracks_data = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None, None

    tracks_data = tracks_data[(tracks_data['popularity'] > 40)]
    cols = ['id', 'name', 'artists'] + features
    tracks_data = tracks_data[cols]
    print(f"Number of tracks before filtering: {len(tracks_data)}")
    tracks_data = tracks_data[tracks_data['name'].apply(is_english_title)]
    print(f"Number of tracks after filtering: {len(tracks_data)}")
    scaler = StandardScaler() if scaler_option == 1 else MinMaxScaler()
    tracks_scaled = scaler.fit_transform(tracks_data[features])
    tracks_sparse = csr_matrix(tracks_scaled)
    
    return tracks_data, tracks_sparse, scaler

def get_song_from_spotify(song_name, artist_name=None):
    try:
        search_query = song_name if not artist_name else f"{song_name} artist:{artist_name}"
        results = sp.search(q=search_query, limit=1, type='track')
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            audio_features = sp.audio_features(track['id'])[0]
            song_details = {
                'id': track['id'],
                'name': track['name'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'explicit': int(track['explicit']),
                'artists': ', '.join([artist['name'] for artist in track['artists']]),
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'key': audio_features['key'],
                'loudness': audio_features['loudness'],
                'mode': audio_features['mode'],
                'speechiness': audio_features['speechiness'],
                'acousticness': audio_features['acousticness'],
                'instrumentalness': audio_features['instrumentalness'],
                'liveness': audio_features['liveness'],
                'valence': audio_features['valence'],
                'tempo': audio_features['tempo'],
                'time_signature': audio_features['time_signature'],
            }
            return song_details
        else:
            return None
    except Exception as e:
        print(f"Error fetching song from Spotify: {e}")
        return None

def update_database_with_song(df, song_info, csv_file_path, scaler, features):
    new_entry = pd.DataFrame([song_info])
    for col in new_entry.columns:
        if col in df.columns:
            new_entry[col] = new_entry[col].astype(df[col].dtype)
    if song_info['id'] not in df['id'].values:
        updated_df = pd.concat([df, new_entry], ignore_index=True)
        updated_df.to_csv(csv_file_path, index=False)
        print("New song added to database.")
    else:
        updated_df = df
        print("Song already exists in the database.")

    # Apply the same scaler to the numerical features of the updated dataframe
    updated_features = updated_df[features]
    scaled_features = scaler.transform(updated_features)
    updated_sparse = csr_matrix(scaled_features)
    print(f"Database updated. Total songs in database: {len(updated_df)}.")
    return updated_df, updated_sparse

def get_close_matches(df, song, artist):
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return pd.DataFrame()

    # Use fuzzy matching for song and artist names
    def is_close_match(row):
        return fuzz.partial_ratio(row['name'].lower(), song.lower()) > 80 and \
               fuzz.partial_ratio(row['artists'].lower(), artist.lower()) > 80

    filtered_df = df[df.apply(is_close_match, axis=1)]

    print(f"Searching for: Song - {song.lower()}, Artist - {artist.lower()}")
    print(f"Found close matches: {filtered_df}")
    return filtered_df



def get_song_details(df, song_name, artist_name=None, csv_file_path=None, scaler=None):
    matched_df = get_close_matches(df, song_name, artist_name)
    if matched_df.empty:
        print("Song not found in the local database. Searching on Spotify...")
        song_info = get_song_from_spotify(song_name, artist_name)
        if song_info:
            print("Song found on Spotify.")
            # In the get_song_details function
            updated_df, updated_sparse = update_database_with_song(df, song_info, csv_file_path, scaler, features)
            global tracks_data, tracks_sparse
            tracks_data, tracks_sparse = updated_df, updated_sparse
            matched_df = get_close_matches(updated_df, song_name, artist_name)
            if matched_df.empty:
                print("No close matches found after updating database.")
                return pd.DataFrame()
        else:
            print("Song not found on Spotify.")
            return pd.DataFrame()
    return matched_df.drop_duplicates(subset='id')
               
def get_recommendations_cosine(song_name, artist, tracks_data, tracks_sparse, top_n=10, csv_file_path=None, scaler=None):
    song_details = get_song_details(tracks_data, song_name, artist, csv_file_path, scaler)
    if song_details.empty:
        print("Debug: No song details found.")
        return pd.DataFrame(columns=['name', 'artists'], data=[['No recommendations available', '']])
    
    track_id = song_details.iloc[0]['id']
    idx_result = np.where(tracks_data['id'] == track_id)[0]
    if idx_result.size == 0:
        print(f"Track ID {track_id} not found in tracks_data.")
        return pd.DataFrame(columns=['name', 'artists'], data=[['No recommendations available', '']])
    
    idx = idx_result[0]
    track_vector = tracks_sparse[idx]
    sim_scores = cosine_similarity(track_vector, tracks_sparse).flatten()
    sim_scores[idx] = -1
    indices = np.argsort(sim_scores)[-top_n:][::-1]

    recommended_track_ids = tracks_data['id'].iloc[indices]
    recommended_tracks = tracks_data.loc[tracks_data['id'].isin(recommended_track_ids), ['id', 'name', 'artists']]
    return recommended_tracks

csv_file_path = 'tracks.csv'
tracks_data, tracks_sparse, scaler = read_data(csv_file_path, scaler_option=0)


# Gradio interface function
def recommend(song_name, artist, top_n=10):
    if tracks_data is not None and tracks_sparse is not None:
        recommendations = get_recommendations_cosine(song_name, artist, tracks_data, tracks_sparse, top_n, csv_file_path, scaler)
        return recommendations[['name', 'artists']]
    else:
        return pd.DataFrame(columns=['name', 'artists'], data=[['Data not available', '']])

# Gradio interface setup
iface = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Textbox(label="Song Name", placeholder="Enter Song Name Here"),
        gr.Textbox(label="Artist", placeholder="Enter Artist Name Here"),
        gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Number of Recommendations")
    ],
    outputs=gr.Dataframe(label="Recommended Songs")
)

iface.launch(share=True)
