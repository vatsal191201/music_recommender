import gradio as gr
from fuzzywuzzy import process
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Initialize the Spotify client
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=os.environ['MY_SECRET_KEY'],
    client_secret=os.environ['MY_SECRET_KEY']))

tracks_data = pd.read_csv('filtered_songs.csv')
df = tracks_data[(tracks_data['popularity'] > 40) & (tracks_data['instrumentalness'] <= 0.85)]

features = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
default_weights = [1/len(features)] * len(features)

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
def calculate_weighted_cosine_similarity(input_song_name, weights, num_songs_to_output, scaler=0):
    input_song = df[df['name'].str.lower() == input_song_name.lower()]

    if input_song.empty:
        print(f"Song named '{input_song_name}' not found in the database.")
        return pd.DataFrame()

    weights = np.array(weights) / np.sum(weights)
    features = df.select_dtypes(include=[np.number])

    if scaler == 0:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    elif scaler == 1:
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = features

    weighted_features = scaled_features * weights
    sparse_weighted_features = csr_matrix(weighted_features)

    input_song_index = input_song.index[0]
    cosine_similarities = cosine_similarity(sparse_weighted_features[input_song_index], sparse_weighted_features).flatten()
    similar_song_indices = np.argsort(-cosine_similarities)[1:num_songs_to_output+1]  # Exclude the first one (input song)
    similar_songs = df.iloc[similar_song_indices][['name', 'artists']]

    return similar_songs

# Function to get the closest match for fuzzy matching
def get_closest_match(query, choices, limit=1):
    result = process.extractOne(query, choices)
    return result[0] if result else None

#description = "Enter a song name and artist name (optional) to get song recommendations. Adjust the feature weights so that they sum up to 1. The system will automatically normalize the weights."


# Function to be called by the Gradio interface
def recommend_songs_interface(song_name, artist_name, num_songs_to_output, scaler_choice, *input_weights):
    num_songs_to_output = int(num_songs_to_output)
    
    # Normalize the weights
    input_weights = np.array(input_weights, dtype=float)
    input_weights /= input_weights.sum()

    # Use default weights if all input weights are zero
    if np.all(input_weights == 0):
        input_weights = np.array(default_weights)

    # Map scaler choice to appropriate scaler
    scaler_map = {"No Scaling": 2, "Standard Scaler": 0, "MinMax Scaler": 1}
    scaler = scaler_map[scaler_choice]

    # Fuzzy match song name and artist
    if artist_name.strip():
        closest_match = get_closest_match(f"{song_name} {artist_name}", df['name'].tolist())
    else:
        closest_match = get_closest_match(song_name, df['name'].tolist())

    # Calculate weighted cosine similarity
    if closest_match:
        similar_songs_df = calculate_weighted_cosine_similarity(closest_match, input_weights, num_songs_to_output, scaler)
        return similar_songs_df
    else:
        return f"No close match found for song '{song_name}'. Please check the spelling and try again."


description = "Enter a song name and artist name (optional) to get song recommendations. Adjust the feature weights using the sliders. The system will automatically normalize the weights."

inputs = [
    gr.components.Textbox(label="Song Name", placeholder="Enter a song name..."),
    gr.components.Textbox(label="Artist Name (optional)", placeholder="Enter artist name (if known)..."),
    gr.components.Number(label="Number of Songs to Output", value=5),
    gr.components.Dropdown(choices=["No Scaling", "Standard Scaler", "MinMax Scaler"], label="Select Scaler", value="No Scaling")
]

# Add sliders for each feature weight
for feature in features:
    inputs.append(gr.components.Slider(minimum=0, maximum=1, value=1/len(features), label=f"Weight for {feature}"))

iface = gr.Interface(
    fn=recommend_songs_interface,
    inputs=inputs,
    outputs=gr.components.Dataframe(),
    title="Song Recommender",
    description=description
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()