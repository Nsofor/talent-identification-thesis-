import tweepy

# Set up your Tweepy credentials
consumer_key = "wS26e43o2MAWo2USDnYUclI6G"
consumer_secret = "TSRw1jUIGWF4uf3aSPZ7sB3WoeNKhmZQiZO2g4HIUjWDj1KGi0"
access_token = "2221784624-Phb9u4joz89etHIoR7LLLM9tKIzOjpWC4BKjU8L"
access_token_secret = "1SwhM1hKz3pwjFzkUb1XonBczwlrp2jVQoQJMCDJc2tGU"

# Authenticate with Twitter's API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
