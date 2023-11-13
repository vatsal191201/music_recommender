README.md for Song Recommender App

Overview

The Song Recommender App is a sophisticated tool designed to provide song recommendations based on a user's input of a song name and optional artist name. The app utilizes a combination of Spotify's API, machine learning techniques, and an intuitive Gradio interface to deliver a personalized music discovery experience.
There were several iterations and technqiues used, ultimately the best performing one can be found in the file called final_recommender_ap.py. 

Live demo:

https://huggingface.co/spaces/Clonkz/music_recommender


Features

Spotify Integration: Fetches song details and audio features directly from Spotify.
Enhanced Fuzzy Matching: Improves song matching accuracy, even with partial or inexact song titles.
Weighted Feature Analysis: Allows users to adjust the importance of various song features such as popularity, danceability, energy, etc.
Customizable Output: Users can specify the number of songs to be recommended.
Gradio Interface: Easy-to-use web interface with sliders and input fields for user interaction.


Installation

To set up and run the Song Recommender App, follow these steps:

Prerequisites
Python 3.10 or higher
Pip package manager
Access to Spotify API (Client ID and Client Secret)
Installation Steps
Clone the repository to your local machine.
Install required Python packages:
Copy code
pip install -r requirements.txt
Set up Spotify API credentials:
Create a .env file in the project root.
Add your Spotify Client ID and Secret as environment variables:
arduino
Copy code
sp_client_id = 'your_spotify_client_id'
sp_client_secret = 'your_spotify_client_secret'
Running the App
Execute the following command in the project directory:

Copy code
python app.py
The Gradio interface will be accessible in your web browser.

Usage

Enter a song name and optionally the artist's name in the provided text boxes.
Adjust the feature weights using the sliders to influence the recommendation criteria.
Choose the number of songs to be recommended.
Select a scaler (Standard Scaler or MinMax Scaler) for feature normalization.
Click 'Submit' to get your personalized song recommendations.
Contributing

Contributions to the Song Recommender App are welcome! Please feel free to fork the repository, make changes, and submit a pull request.

License

This project is open source and available under the MIT License.

Acknowledgements

Spotify API for providing song data and features.
Gradio for the interactive interface framework.

Please note, I don't own any of the data. I merged multiple datasets from Kaggle for this project.
