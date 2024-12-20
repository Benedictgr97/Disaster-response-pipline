import sys
import pandas as pd
import numpy as np 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
    messages_filepath - filpath of the csv messages file
    categories_filepath - filpath of the csv categories file

    OUTPUT:
    df - merged messages and categories 
    
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = categories.merge(messages,how = 'inner',on = 'id')
    return df


def clean_data(df):
    '''
    Clean and one hot encodes categories of the input data frame for category predictions

    INPUT:
    df - dataframe of messages and categories

    OUTPUT: 
    df - dataframe where output categories have been cleaned up and one hot encoded 
    '''



    categories_col = df['categories'].str.split(';',expand = True)
    row = categories_col.head(1)
    category_colnames = list(row.apply(lambda x: x.str.split('-').str[0],axis = 1).loc[0])
    categories_col.columns = category_colnames

    for column in categories_col:
        # set each value to be the last character of the string
        categories_col[column] = categories_col[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories_col[column] =  categories_col[column].astype('int')

    df.drop('categories',axis = 1,inplace = True)
    df = pd.concat([df,categories_col],axis = 1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):

    """
    saves dataframe to an sqllite database.

    INPUT:
    df - the dataframe to be saved.
    database_filename - The path to the sqllite database
    """
    
    engine = create_engine('sqlite:///response_db.db')
    table= database_filename.replace(".db","") + "_table"
    df.to_sql(table, engine, index=False, if_exists = 'replace' ) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()