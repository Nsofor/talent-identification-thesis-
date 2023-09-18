from flask import Flask

app = Flask(__name__)

# Configuration settings for your app (if any)

# Import views and models
from website import views
from website import models
#from website import predalgo
#from website import twitter_api