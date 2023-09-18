from flask import render_template, request, jsonify
import requests
from sklearn.model_selection import train_test_split
from website import app
#from predalgo import train_random_forest_model, load_and_preprocess_data
#from twitter_api import api 
import tweepy
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
from .predalgo import train_random_forest_model, load_and_preprocess_data
from .twitter_api import api
import spacy
from flask import Blueprint

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

bearer_token = "AAAAAAAAAAAAAAAAAAAAAHwBnAEAAAAAicMoesXzGhvoPuOrOVLeygAzajE%3DUacc8tELonO4BVsABI1Q3HO6CJpiZWJZtNKsQYOeV8xZ4eTEHF"


# Replace these values with your actual RapidAPI key and host
RAPIDAPI_KEY = "2cf82eeba7mshab6a66b6027c64dp17b449jsnf37d17d1cc42"
RAPIDAPI_HOST = "sportscore1.p.rapidapi.com"

#prediction
# Load your dataset
df = pd.read_csv('final_dataset.csv')  # Replace 'your_dataset.csv' with your dataset file path

# Assuming 'mbti_personality' is the target column, and other columns are features
X = df[['description', 'followers_count', 'friends_count', 'favourites_count', 'total_favorite_count', 'total_mentions_count', 'total_media_count']]
y = df['mbti_personality']

# Handle missing values in the 'description' column by replacing with an empty string
X['description'].fillna('', inplace=True)

# Text preprocessing: TF-IDF vectorization for 'description' column
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust the number of features
X_description = tfidf_vectorizer.fit_transform(X['description']).toarray()

# Combine TF-IDF vectors with numeric features
X_numeric = X.drop(columns=['description']).values
X_combined = np.hstack((X_numeric, X_description))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

@app.route('/', methods=['GET', 'POST'])
def introduction():
    return render_template('introduction.html')
#def home():
    #stats_data = None
    #error_message = None

    #if request.method == 'POST':
        #player_name = request.form.get('player_name')

        # Step 1: Search for the player by name
        #search_url = 'https://sportscore1.p.rapidapi.com/players/search'
        #search_params = {
          #  'sport_id': '1',  # Assuming '1' is the sport ID for football
         #   'name': player_name
        #}

        #search_headers = {
          #  'X-RapidAPI-Key': RAPIDAPI_KEY,
         #   'X-RapidAPI-Host': RAPIDAPI_HOST
        #}

        #search_response = requests.get(search_url, params=search_params, headers=search_headers)

        #if search_response.status_code == 200:
            #player_info = search_response.json()['data'][0] # Assuming you want the first result
            #player_id = player_info['id']

            # Step 2: Use the player ID to get player statistics
            #stats_url = f'https://sportscore1.p.rapidapi.com/players/{player_id}/statistics'
           # stats_headers = {
          #      'X-RapidAPI-Key': RAPIDAPI_KEY,
         #       'X-RapidAPI-Host': RAPIDAPI_HOST
        #    }

       #     stats_response = requests.get(stats_url, headers=stats_headers)

      #      if stats_response.status_code == 200:
     #           stats_data = stats_response.json()['data']
    #        else:
   #             error_message = "Error retrieving player statistics."
  #      else:
 #           error_message = "Player not found."

    
  

#    return render_template('home.html', stats_data=stats_data, error_message=error_message)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Handle GET request logic here (e.g., render a form)
        return render_template('prediction_form.html')  # Replace with your template name

    if request.method == 'POST':
        # Get user inputs from the form
        description = request.form['description']
        followers_count = int(request.form['followers_count'])
        friends_count = int(request.form['friends_count'])
        favourites_count = int(request.form['favourites_count'])
        total_favorite_count = int(request.form['total_favorite_count'])
        total_mentions_count = int(request.form['total_mentions_count'])
        total_media_count = int(request.form['total_media_count'])

        # Preprocess user inputs
        description_tfidf = tfidf_vectorizer.transform([description]).toarray()
        user_input = np.array([[followers_count, friends_count, favourites_count,
                                total_favorite_count, total_mentions_count, total_media_count]])
        user_input_combined = np.hstack((user_input, description_tfidf))

        # Make predictions
        predicted_personality = rf_classifier.predict(user_input_combined)[0]

        # Prepare the response message
        response_message = f"Predicted Personality: {predicted_personality}"

        # Return the predicted personality as a response
        return response_message


#@app.route('/analyze_tweets', methods=['GET', 'POST'])
#def analyze_tweets():
 #   if request.method == 'POST':
        # Get the search query from the form
  #      search_query = request.form.get('search_query')

        # Search for the latest tweets based on the provided query
   #     query = f"#{search_query}"
    #    tweets = tweepy.Cursor(api.search_tweets, q=query, result_type="recent", count=10).items()

        # Initialize a list to store the extracted names
     #   extracted_names = []

        # Extract names mentioned in the latest tweets
      #  for tweet in tweets:
       #     doc = nlp(tweet.text)
        #    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
         #   extracted_names.extend(names)

        #if extracted_names:
         #   # If names were found in the tweets, display them
          #  return render_template('name_extraction_result.html', names=extracted_names)
        #else:
            # If no names were found, display a message
         #   return render_template('name_extraction_result.html', message="No names mentioned in the tweets.")

   # return render_template('name_extraction_form.html')


@app.route('/player_id_form', methods=['GET'])
def player_id_form():
    return render_template('player_id_form.html')

@app.route('/player_id', methods=['GET', 'POST'])
def get_player_id():
    if request.method == 'POST':
        # Extract sport_id and name from the form submitted by the user
        sport_id = request.form.get('sport_id')
        player_name = request.form.get('player_name')

        # Define the URL for the API endpoint
        url = "https://sportscore1.p.rapidapi.com/players/search"

        # Define the query parameters
        query_params = {
            "sport_id": sport_id,
            "name": player_name
        }

        # Define the headers including your RapidAPI key and host
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": RAPIDAPI_HOST
        }

        # Make a POST request to the API
        response = requests.post(url, headers=headers, params=query_params)

        if response.status_code == 200:
            # Parse the JSON response
            data = response.json().get('data', [])

            if data:
                # Extract the ID of the first player in the response
                player_id = data[0].get('id')
                return f"Player ID for {player_name}: {player_id}"
            else:
                return "Player not found."
        else:
            return "Error retrieving player data."

    return "This route is meant to handle POST requests only."


@app.route('/player_stats/<int:player_id>', methods=['GET'])
def get_player_stats(player_id):
    # Define the URL for the API endpoint
    url = f"https://sportscore1.p.rapidapi.com/players/{player_id}/statistics"

    # Define the query parameters (if needed)
    query_params = {"page": "1"}

    # Define the headers including your RapidAPI key and host
    headers = {
        "X-RapidAPI-Key": "2cf82eeba7mshab6a66b6027c64dp17b449jsnf37d17d1cc42",
        "X-RapidAPI-Host": "sportscore1.p.rapidapi.com"
    }

    # Make a GET request to the API
    response = requests.get(url, headers=headers, params=query_params)

    if response.status_code == 200:
        # Parse the JSON response
        data = response.json().get('data', [])
        
        if data:
            # You can process the player statistics data here and render it in a template
            #print(data)
            return render_template('player_stats.html', player_stats=data)
        else:
            return "Player statistics not found."
    else:
        return "Error retrieving player statistics."


@app.route('/search_tweets', methods=['GET', 'POST'])
def search_tweets():
    if request.method == 'POST':
        search_query = request.form.get('search_query', '#u21euro talent -is:retweet')
        num_tweets = int(request.form.get('num_tweets', 10))

        # Create headers with the Bearer Token
        headers = {
            "Authorization": f"Bearer {bearer_token}"
        }

        # Define the Twitter API endpoint
        endpoint_url = "https://api.twitter.com/2/tweets/search/recent"

        # Define the query parameters
        params = {
            "query": search_query,
            "max_results": num_tweets,
            "tweet.fields": "text",  # Include tweet text in the response
            "expansions": "author_id",  # Expand author_id to get user details
            "user.fields": "username"  # Include user username in the response
        }

        # Make the API request
        response = requests.get(endpoint_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            tweets = data.get("data", [])

            # Extract names using NER
            names = []

            for tweet in tweets:
                tweet_text = tweet.get("text", "")
                doc = nlp(tweet_text)

                # Extract PERSON entities (names)
                extracted_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

                # Add the extracted names to the list
                names.extend(extracted_names)

            # Remove duplicates
            unique_names = list(set(names))

            return render_template('search_tweets.html', names=unique_names)

    return render_template('search_tweets.html', names=[])

    

