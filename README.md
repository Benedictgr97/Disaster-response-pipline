# Disaster Response Pipeline Project

### Description:
A pipline that ingests disaster response messages and, transforms the message and runs it through a 'bag of words' NLP model approach to classify these messages to a given type of disaster.

### Installation and requirements:
Python version : 3.12.4

To gain the access the required packages, run the command below whilst in the same location as the requierments.txt folder:

```bash pip install -r requirements.txt ```

$ git clone https://github.com/Swatichanchal/Disaster-Response-Pipeline.git
In addition This will require pip installation of the following:

$ pip install SQLAlchemy
$ pip install nltk
Python 3+
ML Libraries: NumPy, Pandas, SciPy, SkLearn
NLP Libraries: NLTK
SQLlite Libraries: SQLalchemy
Model Loading and Saving Library: Pickle
Web App and Visualization: Flask, Plotly
The code can be viewed and modified with Jupyter Notebooks.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


