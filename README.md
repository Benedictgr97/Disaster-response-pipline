# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Installation and requirements](#installation_and_requirements)
3. [Instructions](#instructions)
4. [Additional files](#additional_files)
5. [Example](#example)
6. [Acknowledgements](#acknowledgements)

<a name="descripton"></a>
## Description:
A pipline that ingests disaster response messages and, transforms the message and runs it through a 'bag of words' NLP model approach to classify these messages to a given type of disaster.

There are three main sections of this script, the first two are for set up and the final one is for running the model predictions alongisde displaying the layout of the data.

1. **Process_Data.py**: ETL on the input data, storing it in a temporary SQLite database ready for the model.
2. **Train_Classifier.py**: Used to train and evaluate the model from the transformed data ready for future predictions.
3. **Run.py**: Displays the layout of the training data in a web app and allows the user to predict the disaster category of a new response.

<a name="installation_and_requirements"></a>
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
<a name="instructions"></a>
## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Response_db.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/response_db.db models/XGB_pipeline.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="additional_files"></a>
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

<a name="example"></a>
## **Example**: _Running each part_
1. Run the ETL pipline, if you have new data to train on, update the data/disaster_messages.csv data/disaster_categories.csv

![image](https://github.com/user-attachments/assets/2f92fce7-1c7a-4c8b-b975-18d0c05482a1)

2. Run the ML pilone that trains the classifier
   
![image](https://github.com/user-attachments/assets/04ac348e-5c45-49cc-87ac-25745e74ae63)
![image](https://github.com/user-attachments/assets/2aa41294-5595-4cce-b1a2-8fc95d841580)

3. Running run.py whilst in the app folder will give the below in browser

![image](https://github.com/user-attachments/assets/b80f1f40-7e78-4b18-a820-acbca38146a9)

4.Finally, entering a disaster response message to clarify will give the below

![image](https://github.com/user-attachments/assets/6d0ebe61-2015-4fa5-8d2f-a81dfd88613a)

<a name="acknowledgements"></a>
## Acknowledgements
- [Udacity](https://www.udacity.com/) : Providing the templates, outline and training for this course.
- [Figure Eight](https://www.appen.com/): Providing the disaster response data.




