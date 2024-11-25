# Disaster Response Pipeline Project

## Description:
A pipline that ingests disaster response messages and, transforms the message and runs it through a 'bag of words' NLP model approach to classify these messages to a given type of disaster.

There are three main sections of this script, the first two are for set up and the final one is for running the model predictions alongisde displaying the layout of the data.

1. **Process_Data.py**: ETL on the input data, storing it in a temporary SQLite database ready for the model.
2. **Train_Classifier.py**: Used to train and evaluate the model from the transformed data ready for future predictions.
3. **Run.py**: Displays the layout of the training data in a web app and allows the user to predict the disaster category of a new response.

## Installation and requirements:
Python version : 3.12.4

To gain the access the required packages, run the command below whilst in the same location as the requierments.txt folder:

```
pip install -r requirements.txt
```

Run the below command to clone the repository:

```
git clone https://github.com/Benedictgr97/Disaster-response-pipline.git
``` 

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Response_db.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/response_db.db models/XGB_pipeline.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Additional files:

**Folder** - _app_
- **run.py** - Described above
- **Folder** -  _templates_ - HTML templates for the web application. 

**Folder** - _data_
- **ETL Pipeline Preparation.ipynb** - Experimentation to build out the process_data.py pipline with methodology included 
- **Response_db.db** - SQLlite Database created from process_data.py containing transformed data
- **disaster_categories.csv** - Categories for each disaster message 
- **disaster_messages.csv** - Real disater messages using for training the classification model
- **process_data.py** - Described above

**Folder** - _models_
- **ML Pipeline Preparation.ipynb** - Experimentation to build out the train_classifier.py model with methodology included 
- **XGB_pipeline.pkl** - Pickle file of XGB boost trained model
- **train_classifier.py** -  Described above

## **Example**: _Training the model_
2. Run the ML pilone that trains the classifier
   
![image](https://github.com/user-attachments/assets/04ac348e-5c45-49cc-87ac-25745e74ae63)



## Acknowledgements
- [Udacity](https://www.udacity.com/) : Providing the templates, outline and training for this course.
- [Figure Eight](https://www.appen.com/): Providing the disaster response data.




