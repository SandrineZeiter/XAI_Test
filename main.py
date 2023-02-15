# imports for flask
import flask
import json
import os
from flask import send_from_directory, request, Flask

# imports for ML and LIME
import random
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
matplotlib.use("Agg") #To save the figure
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('axes', labelsize=16)
matplotlib.rc('figure', titlesize=20)

textToAnalyze = " "
# ----------------------- ML-part -----------------------
#def machine_learning():
# import dataset
data = pd.read_csv("/Users/sandrinezeiter/Library/CloudStorage/OneDrive-UniversitédeFribourg/"
                   "Thesis/Twitter_cleaned.csv")

# define the different class names for feelings
class_names = ['afraid', 'alive', 'angry', 'confused', 'depressed', 'good', 'happy',
          'helpless', 'hurt', 'indifferent', 'interested', 'love', 'open', 'positive',
          'sad', 'strong']

# split into training and test set
train, test = train_test_split(data, train_size=0.9)

# tokenize the sentences
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\b[a-zA-Z]{3,}\b', lowercase=False,
                        min_df=5, max_df=0.7, stop_words='english')
vectorizer.fit_transform(train['tweet'])

mnb = MultinomialNB(alpha=0.1)
p1 = make_pipeline(vectorizer, mnb)

alpha_grid = np.logspace(-3, 0, 4)  # Is smoothing parameter for the counts
param_grid = [{'multinomialnb__alpha': alpha_grid}]
gs = GridSearchCV(p1, param_grid=param_grid, cv=5, return_train_score=True)

gs.fit(train.tweet, train.feeling)

# ----------------------- LIME -----------------------

def lime_testing_userinput(userinput):

    explainer = LimeTextExplainer(class_names=class_names)
    p1.fit(train.tweet, train.sub_category)
    exp2 = explainer.explain_instance(userinput,
                                      p1.predict_proba,
                                      num_features=5,
                                      top_labels=len(class_names))

    print(userinput)
    class_index = exp2.available_labels()[0]
    #print("Class index ", class_index)

    prediction = class_names[class_index]
    print("Prediction ", prediction)
    exp2.show_in_notebook([class_index])
    exp2.as_pyplot_figure(label=exp2.available_labels()[0])
    plt.savefig("figure.png", bbox_inches="tight")
    #plt.show()

    print('Explanation for class %s' % class_names[class_index])
    print('\n'.join(map(str, exp2.as_list(label=class_index))))

    return prediction

def lime_testing():
    idx = random.randint(1, len(test.tweet))
    # idx = 1

    explainer = LimeTextExplainer(class_names=class_names)
    p1.fit(train.tweet, train.sub_category)
    exp = explainer.explain_instance(test.tweet.iloc[idx],
                                     p1.predict_proba,
                                     num_features=5,
                                     # labels=[0,15],
                                     top_labels=2)

    print(test['tweet'].iloc[idx])

    print('Document id: %d' % idx)
    print('True class: %s' % test.sub_category.iloc[idx])
    print('R2 score: {:.3f}'.format(exp.score))
    exp.show_in_notebook(text=True)

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
    global textToAnalyze

    req = request.get_json(silent=True, force=True)

    fulfillmentText = ''
    query_result = req.get('queryResult')
    #print(query_result)

    #fulfillmentText = "it worked ! "
    if query_result.get('action') == 'get.address': # if intent oou action is equal ...
        ### Perform set of executable code
        ### if required
        ### ...

        fulfillmentText = "got address"

    elif query_result.get('action') == 'get.feeling':
        fulfillmentText = "got feeling"
        lime_testing()

    elif query_result.get('action') == 'fallback':
        userinput = query_result["queryText"]
        textToAnalyze = textToAnalyze + " " + userinput
        if len(textToAnalyze) < 64:
            # prediction = lime_testing_userinput(userinput)
            # fulfillmentText = "Thank you for telling me about your day. According to what you said, you feel " + prediction + "."
            fulfillmentText = "Can you tell me more about that?"
            print("Text too short")
            #textToAnalyze = textToAnalyze + " " + userinput
            print("Text to analyze: ", textToAnalyze)
            print("end of input")
            # print("User input 1", userinput)
        else:
            prediction = lime_testing_userinput(textToAnalyze)
            print("else")
            fulfillmentText = "Thank you for telling me about your day. According to what you said, you feel " + prediction + "."
            #fulfillmentText = "Thank you for telling me about your day. According to what you said, you feel " + prediction + "."
            textToAnalyze = '' #So that I don't have to restart the whole script over and over again.

    elif query_result.get('action') == 'get.informationone':

        userinput = query_result["queryText"]
        textToAnalyze = textToAnalyze + " " + userinput
        if len(textToAnalyze) < 256:
            #prediction = lime_testing_userinput(userinput)
            #fulfillmentText = "Thank you for telling me about your day. According to what you said, you feel " + prediction + "."
            fulfillmentText = "Can you tell me more about that?"
            print("Text too short")
            textToAnalyze = textToAnalyze + " " + userinput
            print("Text to analyze", textToAnalyze)
            #print("User input 1", userinput)
        else:
            prediction = lime_testing_userinput(textToAnalyze)
            print("else")
            fulfillmentText = "Thank you for telling me about your day. According to what you said, you feel " + prediction + "."


    elif query_result.get('action') == "get.informationtwo":
        userinput = query_result["queryText"]
        textToAnalyze = textToAnalyze + " " + userinput
        prediction = lime_testing_userinput(textToAnalyze)
        fulfillmentText = "Thank you for telling me about your day. According to what you said, you feel " + prediction + "."
        #fulfillmentText = "Thank you."
        #print("Text to analyze 2", textToAnalyze)
        #print("User input 2", userinput)


    elif query_result.get('action') == 'input.unknown':
        #print(query_result["queryText"])
        lime_testing_userinput(query_result["queryText"])
        fulfillmentText = "Got the information after lime"

    return {
        "fulfillmentText": fulfillmentText,
        "source": "webhookdata"
    }


# run the app
if __name__ == '__main__':
    app.run()