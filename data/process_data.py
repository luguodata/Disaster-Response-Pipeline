import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load message data and categories datasets for DisasterResponse project

    Args:
        messages_filepath(str): file path of message data
        categories_filepath(str): file path of categories data

    Returns:
        df: Merged dataset of message and categories
    """
    messages = pd.read_csv(messages_filepath, sep = ',')
    categories = pd.read_csv(categories_filepath, sep = ',')
    df = messages.merge(categories, on = ['id'], how = 'left')

    return df


def clean_data(df):
    """ Clean merge data of message and categories. This step includes
        categories splits, converting categories into dummy values, drop
        duplicated records.

    Args:
        df: merged data from message and categories

    Returns:
        df: cleaned df
    """
    categorie_seg = df['categories'].str.split(';', expand = True)
    category_colnames = categorie_seg.head(1).values[0].tolist()
    categorie_seg.columns = [col[:-2] for col in category_colnames]

    for column in categorie_seg:
        # set each value to be the last character of the string
        categorie_seg[column] = categorie_seg[column].str[-1]

        # convert column from string to numeric
        categorie_seg[column] = categorie_seg[column].astype(int)

    # drop original categories column
    df = df.drop(['categories'], axis = 1)
    # merge message and categories data together
    df = pd.concat([df, categorie_seg], axis = 1)

    # move related - 1  's value 2 into 0
    df.loc[df['related'] == 2, 'related'] = 0

    # remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Create sqlite database and save the processed data into it.

    Args:
        df: cleaned merging data of message and categories
        database_filename: sqlite database name

    Return:
        None
    """

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('etl_processed_data', engine, index=False)


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
