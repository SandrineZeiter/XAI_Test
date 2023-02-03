# imports for flask
import flask
import json
import os
from flask import send_from_directory, request, Flask

# imports for ML and LIME
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lime.lime_text import LimeTextExplainer
from lime import lime_text

import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('axes', labelsize=16)
matplotlib.rc('figure', titlesize=20)

# import dataset
data = pd.read_csv("/Users/sandrinezeiter/Library/CloudStorage/OneDrive-UniversitédeFribourg/"
                       "Thesis/Twitter_cleaned.csv")

# define the different class names for feelings
class_names = ['afraid', 'alive', 'angry', 'confused', 'depressed', 'good', 'happy',
              'helpless', 'hurt', 'indifferent', 'interested', 'love', 'open', 'positive',
              'sad', 'strong']

#

# initialize the flask app
app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return "Hello World"
# create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])

# define webhook for different actions and results
def webhook():

    req = request.get_json(silent=True, force=True)

    fullfillmentText = ''
    query_result = req.get('queryResult')
    fulfillmentText = "it worked ! "
    if query_result.get('action') == 'get.address': # if intent oou action is equal ...
        ### Perform set of executable code
        ### if required
        ### ...

        fulfillmentText = "got address"

    elif query_result.get('action') == 'get.feeling':
        fulfillmentText = "got feeling"

    return {
        "fulfillmentText": fulfillmentText,
        "source": "webhookdata"
    }


# run the app
if __name__ == '__main__':
    app.run()