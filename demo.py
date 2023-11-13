import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from fuzzywuzzy import process, fuzz

# Spotify API setup
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id='76ff1108754147b0b59b9beb48e6af37',
    client_secret='32a408639a054cfead2b0432ac5b1f59'))

# Define features for scaling and calculations
features = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
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

# Function to calculate weighted cosine similarity
def calculate_weighted_cosine_similarity(input_song_name, weights, num_songs_to_output, scaler_choice, tracks_data):
    input_song = tracks_data[tracks_data['name'].str.lower() == input_song_name.lower()]

    if input_song.empty:
        print(f"Song named '{input_song_name}' not found in the database.")
        return pd.DataFrame()

    # Select only the intended features for scaling
    selected_features = tracks_data[features]  

    # Your existing code for scaling
    if scaler_choice == 0:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(selected_features)
    elif scaler_choice == 1:
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(selected_features)
    else:
        scaled_features = selected_features.values  # Convert DataFrame to NumPy array

    # Ensure the weights match the number of features
    weighted_features = scaled_features * weights.reshape(1, -1)
    sparse_weighted_features = csr_matrix(weighted_features)

    input_song_index = input_song.index[0]
    cosine_similarities = cosine_similarity(sparse_weighted_features[input_song_index], sparse_weighted_features).flatten()
    similar_song_indices = np.argsort(-cosine_similarities)[1:num_songs_to_output+1]  # Exclude the first one (input song)
    similar_songs = tracks_data.iloc[similar_song_indices][['name', 'artists']]  # Corrected from df to tracks_data

    return similar_songs

# Function to get the closest match for fuzzy matching
def get_closest_match(query, choices, limit=1):
    result = process.extractOne(query, choices)
    return result[0] if result else None

# Function to recommend songs
def recommend_songs_interface(song_name, artist_name, num_songs_to_output, scaler_choice, tracks_data, *input_weights):
    num_songs_to_output = int(num_songs_to_output)
    weights = [float(weight) for weight in input_weights]  # Ensure this conversion is done correctly

    # Normalize the weights
    weights = np.array(weights)
    weights /= np.sum(weights)

    # Use default weights if all input weights are zero
    if np.all(weights == 0):
        weights = np.array(default_weights)

    # Map scaler choice to appropriate scaler
    scaler_map = {"No Scaling": 2, "Standard Scaler": 0, "MinMax Scaler": 1}
    scaler_choice = scaler_map[scaler_choice]

    # Fuzzy match song name and artist
    closest_match = get_closest_match(f"{song_name} {artist_name}", tracks_data['name'].tolist()) if artist_name.strip() else get_closest_match(song_name, tracks_data['name'].tolist())

    # Calculate weighted cosine similarity
    if closest_match:
        similar_songs_df = calculate_weighted_cosine_similarity(closest_match, weights, num_songs_to_output, scaler_choice, tracks_data)
        return similar_songs_df
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
