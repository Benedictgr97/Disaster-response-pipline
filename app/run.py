import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

import pickle # Joblib is failing to pull out tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Pie
import plotly.express as px
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator,TransformerMixin


app = Flask(__name__)

#This class identifies if the first word of each sentence in a text is a verb or ‘RT’ 
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    """
    Used in Piplines to see if the first word of the sentance is a verb
    
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if pos_tags:  # Check if pos_tags is not empty
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
database_filepath = "../data/response_db.db"
engine = create_engine('sqlite:///' + database_filepath)
df = pd.read_sql_table('Response_db_table', engine)

# load model
with open("../models/XGB_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
@app.route('/index')
def index():
    # Extract breakdown of genre
    genre_counts = df.groupby('genre').count()['message']
    category_names = df.iloc[:, 4:].columns
    category_total = (df.iloc[:, 4:] != 0).sum().values

    # Sort the data
    sorted_genre_counts = genre_counts.sort_values(ascending=False)
    sorted_genre_names = list(sorted_genre_counts.index)

    sorted_category_indices = category_total.argsort()[::-1]
    sorted_category_total = category_total[sorted_category_indices]
    sorted_category_names = category_names[sorted_catsegory_indices]

    # Create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=sorted_genre_names,
                    values=sorted_genre_counts
                )
            ],
            'layout': {
                'title': 'Breakdown of categories of received messages'
            }
        },
        {
            'data': [
                Bar(
                    x=sorted_category_names,
                    y=sorted_category_total
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()