# Disaster Response Pipeline Project

### Description:
A pipline that ingests disaster response messages and, transforms the message and runs it through a 'bag of words' NLP model approach to classify these messages to a given type of disaster.

There are three main sections of this script, the first two are for set up and the final one is for running the model predictions alongisde displaying the layout of the data.

1. **Process_Data.py**: ETL on the input data, storing it in a temporary SQLite database ready for the model.
2. **Train_Classifier.py**: Used to train and evaluate the model from the transformed data ready for future predictions.
3. **Run.py**: Displays the layout of the training data in a web app and allows the user to predict the disaster category of a new response.

### Installation and requirements:
Python version : 3.12.4

To gain the access the required packages, run the command below whilst in the same location as the requierments.txt folder:

```
pip install -r requirements.txt
```

Run the below command to clone the repository:

```
git clone https://github.com/Benedictgr97/Disaster-response-pipline.git
``` 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Response_db.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/response_db.db models/XGB_pipeline.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Additional files:

**Folder** - _app_
run.py - python script to launch web application.
Folder: templates - web dependency files (go.html & master.html) required to run the web application.

**Folder** - _data_
disaster_messages.csv - real messages sent during disaster events (provided by Figure Eight)
disaster_categories.csv - categories of the messages
process_data.py - ETL pipeline used to load, clean, extract feature and store data in SQLite database
ETL Pipeline Preparation.ipynb - Jupyter Notebook used to prepare ETL pipeline
DisasterResponse.db - cleaned data stored in SQlite database

**Folder** - _models_
train_classifier.py - ML pipeline used to load cleaned data, train model and save trained model as pickle (.pkl) file for later use
classifier.pkl - pickle file contains trained model
ML Pipeline Preparation.ipynb - Jupyter Notebook used to prepare ML pipeline

