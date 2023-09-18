import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

def load_and_preprocess_data():
    # Read the first CSV file
    df1 = pd.read_csv("C:/Users/TOBE/Desktop/flaskwebappproject/website/mbti_labels.csv")

    # Read the second CSV file
    df2 = pd.read_csv("C:/Users/TOBE/Desktop/flaskwebappproject/website/user_info.csv")

    # Merge the datasets based on the 'username' column
    merged_df = pd.merge(df1, df2, on='id', how='left')

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv('merged_dataset.csv', index=False)

    # List of column names to remove
    columns_to_remove = ["average_url_count", "average_mentions_count","location","total_url_count","screen_name","name","id_str"]

    # Drop the unwanted columns from the DataFrame
    merged_df_ = merged_df.drop(columns=columns_to_remove)

    # Specify the new filename for the saved dataset
    new_filename = 'new_merged_dataset.csv'

    # Save the modified DataFrame as a new CSV file
    merged_df_.to_csv(new_filename, index=False)

    # Load the dataset
    data = pd.read_csv('new_merged_dataset.csv')

    # Handle missing values in the 'bio' column
    data['description'].fillna('', inplace=True)

    # Features (X) and target (y)
    X_text = data['description']
    X_numeric = data[['followers_count', 'friends_count','listed_count','favourites_count','statuses_count','number_of_quoted_statuses','number_of_retweeted_statuses','total_retweet_count','total_favorite_count','total_hashtag_count','total_mentions_count','total_media_count','average_tweet_length','average_retweet_count','average_favorite_count','average_hashtag_count','average_media_count']]

    # Preprocess text data using TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_text_tfidf = tfidf_vectorizer.fit_transform(X_text)

    # Concatenate text and numeric features
    X = scipy.sparse.hstack((X_text_tfidf, X_numeric), format='csr')

    y = data['mbti_personality']

    return X, y

def train_random_forest_model(X, y):
    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Random Forest classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    #print("Accuracy:", accuracy)
    #print("Classification Report:\n", classification_rep)

    return rf_classifier
