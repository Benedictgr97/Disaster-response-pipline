"""
Disaster reponse clasification training

Arguments: 
    1. Pickle file output location 

Outputs: 
    1. Pickle file of model

"""


# import libraries
import pandas as pd
import numpy as np 

import re
import sys
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator,TransformerMixin

import pickle

import xgboost as xgb

import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


# Function to suppress output
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Response_db_table',engine)

    # Child alone has 0 for all values so drop 
    # Realted has a max of 2, most likley a fluke for a binary input 
    # Only Nulls in originl and this will be dropped as its not usefull in this context 
    df.drop('child_alone',axis = 1,inplace=True)
    df['related'] = df['related'].apply(lambda x: 1 if x > 0 else 0)

    X = df['message']
    y = df.drop(['id','message','original','genre'],axis = 1)

    category_names = y.columns
    return X,y,category_names


def tokenize(text):
    """
    Tokenize the disaster response text
    
    Arguments: text -> Input text that requires 
    Output: final_tokens -> tokens from the extracted text
    """

    #For all url's, replace with a place holder string
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_vals = re.findall(url_regex,text)
    for url in url_vals:
        text = text.replace(url, 'urlplacehold')

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    stop_words = set(stopwords.words('english'))
    final_tokens= [stemmer.stem(lemmatizer.lemmatize(w).lower().strip()) for w in tokens if w.lower() not in stop_words]

    return final_tokens

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


def build_model():
    """
    Builds a pipline finction with the starting verb extractor and xgboost as the model 
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize,token_pattern=None)),
                ('tfidf_transformer', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('classifier', MultiOutputClassifier(xgb.XGBClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT: 
    Model - ML model
    X_test - Test input messages 
    Y_test - Categories of message
    Category_names - The names of categories for y_test

    OUTPUT:

    Weighted average of the model to make sure we are classifying the most comman repsonses correctly
    """

    Y_predict = model.predict(X_test)

    report = classification_report(Y_test.values, Y_predict, target_names= category_names, output_dict=True, zero_division=0)

    weighted_avg = report['weighted avg']
    print("Weighted Average Metrics:")
    for metric, value in weighted_avg.items():
        print(f"{metric.capitalize()}: {value:.2f}")

    print(classification_report(Y_test.values, Y_predict, target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()