# Disaster Response Pipeline Project

### Description:
A pipline that ingests disaster response messages and, transforms the message and runs it through a 'bag of words' NLP model approach to classify these messages to a given type of disaster.

### Installation and requirements:
Python version : 3.12.4

To gain the access the required packages, run the command below whilst in the same location as the requierments.txt folder:

```
bash pip install -r requirements.txt
```

Run the below command to clone the repository:

```
bash git clone https://github.com/Benedictgr97/Disaster-response-pipline.git
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


