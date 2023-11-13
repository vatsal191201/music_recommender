import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix
from rapidfuzz import process, fuzz
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API setup
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id='',
    client_secret=''))

# Define features for scaling and calculations
features = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
default_weights = [1/len(features)] * len(features)

# Read and preprocess the data
tracks_data = pd.read_csv('filtered_songs.csv')
tracks_data = tracks_data[(tracks_data['popularity'] > 40) & (tracks_data['instrumentalness'] <= 0.85)]

# Function to fetch a song from Spotify
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

# Enhanced Fuzzy Matching Function
def enhanced_fuzzy_matching(song_name, artist_name, df):
    combined_query = f"{song_name} {artist_name}".strip()
    df['combined'] = df['name'] + ' ' + df['artists']
    matches = process.extractOne(combined_query, df['combined'], scorer=fuzz.token_sort_ratio)
    return df.index[df['combined'] == matches[0]].tolist()[0] if matches else None

# Function to apply the selected scaler and calculate weighted cosine similarity
def calculate_weighted_cosine_similarity(input_song_index, weights, num_songs_to_output, tracks_data, scaler_choice):
    # Apply the selected scaler
    if scaler_choice == 'Standard Scaler':
        scaler = StandardScaler()
    else:  # MinMaxScaler
        scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(tracks_data[features]) * weights
    tracks_sparse = csr_matrix(scaled_features)

    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(tracks_sparse[input_song_index], tracks_sparse).flatten()
    similar_song_indices = np.argsort(-cosine_similarities)[1:num_songs_to_output+1]
    return similar_song_indices


# Function to recommend songs
def recommend_songs_interface(song_name, artist_name, num_songs_to_output, scaler_choice, tracks_data, *input_weights):
    num_songs_to_output = int(num_songs_to_output)
    weights = np.array([float(weight) for weight in input_weights]) if input_weights else default_weights
    weights /= np.sum(weights)  # Normalize weights

    song_index = enhanced_fuzzy_matching(song_name, artist_name, tracks_data)
    if song_index is not None:
        similar_indices = calculate_weighted_cosine_similarity(song_index, weights, num_songs_to_output, tracks_data, scaler_choice)
        similar_songs = tracks_data.iloc[similar_indices][['name', 'artists']]
        return similar_songs
    else:
        return pd.DataFrame(columns=['name', 'artists'])

# Gradio interface setup
description = "Enter a song name and artist name (optional) to get song recommendations. Adjust the feature weights using the sliders. The system will automatically normalize the weights."

inputs = [
    gr.components.Textbox(label="Song Name", placeholder="Enter a song name..."),
    gr.components.Textbox(label="Artist Name (optional)", placeholder="Enter artist name (if known)..."),
    gr.components.Number(label="Number of Songs to Output", value=5),
    gr.components.Dropdown(choices=["Standard Scaler", "MinMax Scaler"], label="Select Scaler", value="Standard Scaler")
]

# Add sliders for each feature weight
for feature in features:
    inputs.append(gr.components.Slider(minimum=0, maximum=1, value=1/len(features), label=f"Weight for {feature}"))

# Gradio interface setup
iface = gr.Interface(
    fn=lambda song_name, artist_name, num_songs_to_output, scaler_choice, *input_weights: recommend_songs_interface(song_name, artist_name, num_songs_to_output, scaler_choice, tracks_data, *input_weights),
    inputs=inputs,
    outputs=gr.components.Dataframe(),
    title="Song Recommender",
    description=description
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()
