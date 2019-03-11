# import libraries
import sys
import re

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')



def load_data(database_filepath):
    """ Load dataset from the ETL processed databaseself.Separate for predictors
        variables (X) and target variables (Y). Also, get targets' names list.

    Args:
        database_filepath (str): the saved database's path from ETL processself.

    Return:
        X: predictors subset of ETL processed dataset
        Y: targets subset of ETL processed dataset
        category_names: the list of target variable names.
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('etl_processed_data', engine)

    # split training and testing dataset
    X = df.ix[:,1:2].values[:,0]
    Y = df.ix[:,4:].values

    # get target variable names
    category_names = df.ix[:,4:].columns.tolist()

    return X, Y, category_names




def tokenize(text):
    """Nomalize, tokenize and lemmatize process for text

    Args:
        text (str)

    Return:
        text_lems: list of element of the text after normalization, tokenization
        stopwords removal and lemmetization
    """
    # Normailization -- lower case + remove puntuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())

    # tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    words = [word for word in tokens if word not in stopwords.words("english")]

    # stemmization
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]

    return text_lems


# Comment this part for suitable for app input information
#
# class Cate_Text_Selector(BaseEstimator, TransformerMixin):
#     """The Cate_Text_Selector to identify text variables and normal categorical
#        variable. This process in order to be embeded as custom transformer to
#        process text variables and categorical variables separatly
#     """
#     def __init__(self, dtype):
#         """ Input the data type to be kept after selector's filtering
#             "text" or "category"
#         """
#         self.dtype = dtype
#
#     def fit(self, X, y = None):
#         return self
#
#     def text_selector(self, X):
#         """ Identify text variables and normal categorical variables, then
#             assign them to different listsself.
#             Criteria for identify text variable: Originally was object type and
#             # of unique values greater than # of total non-null values in the
#             dataframe
#
#         Args:
#             X: dataframe which contains need to be split variables.
#
#         Return:
#             text_col: list of text variable names.
#             cate_col: list of normal categorical variable names.
#
#         """
#
#         text_col = []
#         cate_col = []
#
#         for col in X.select_dtypes(include='object').columns.tolist():
#             if len(X[col].unique()) > 0.5 * len(X[X[col].notnull()]):
#                 text_col.append(col)
#             else:
#                 cate_col.append(col)
#
#         return text_col, cate_col
#
#     def transform(self, X):
#         """Transform selected text columns and normal categorical columns into
#            arrays suitable for the following feature transformation and model
#            traning process
#
#         Args:
#             X: dataframe which contains need to be split and transformed
#                variables.
#
#         Return:
#             Arrays of selected type variables.
#         """
#
#         text_col, cate_col = self.text_selector(X)
#
#         if self.dtype == 'text':
#             return X[text_col].values[:,0]
#         if self.dtype == 'category':
#             return X[cate_col].values



# def build_model():
#     """ The model builing process to integrate all the necessary steps of model
#         training, which include data loading, transformation, model training,
#         parameter grid search, model evaluation and save the trained model.
#     """
#
#     # feature preprocessing pipeline
#     pipeline = Pipeline([
#         ('features', FeatureUnion([
#             ('text_features', Pipeline([
#                 ('selector',Cate_Text_Selector('text')),
#                 ('vect', CountVectorizer(tokenizer = tokenize)),
#                 ('tfidf', TfidfTransformer())
#             ])),# Text preprocessing ends
#
#             ('cate_features', Pipeline([
#                 ('selector',Cate_Text_Selector('category')),
#                 ('dummy', OneHotEncoder())
#             ])) # Normal categorical variable preprocessing ends
#         ])), # Feature part ends
#
#         ('clf', MultiOutputClassifier(RandomForestClassifier()))
#     ])
#
#     # grid search
#     parameters = {
#         #'clf__estimator__n_estimators': [20, 50],
#         'clf__estimator__max_depth': [3, 6],
#         'clf__estimator__min_samples_split': [2,4]
#         #'clf__estimator__loss': ['log', 'hinge']
#         #'clf__estimator__penalty': ['l2']
#         #'clf__estimator__alpha': [0.001, 0.0001]
#     }
#
#     cv = GridSearchCV(pipeline, param_grid= parameters)
#
#     return cv



def build_model():
    """ The model builing process to integrate all the necessary steps of model
        training, which include data loading, transformation, model training,
        parameter grid search, model evaluation and save the trained model.
    """

    # feature preprocessing pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # grid search
    parameters = {
        #'clf__estimator__n_estimators': [10,20],
        #'clf__estimator__max_depth': [4, 8],
        'clf__estimator__min_samples_split': [3,6]
    }

    cv = GridSearchCV(pipeline, param_grid= parameters)

    return cv





def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluation trained model via comparing true Y_test values and predicted
        test values

    Args:
        model: the trained model
        X_test: subset dataframe of selected test predictors
        Y_test: subset dataframe of selected test tragets
        category_names: the list of traget variables name list

    Return:
        None. Classification report which includes precison, recall, f1
        score for each target variable, and averaged f1 score of all target
        variables  will be printed out while calling the function.
    """
    # prediction on test dataset
    Y_pred = model.predict(X_test)

    # initialize aggregated avg score
    weighted_fscore = 0

    for i in np.arange(0,36,1):
        print("Target:{}".format(category_names[i]))
        print("\n")
        print(classification_report(Y_test[i],Y_pred[i]))
        print('\n')
        print('\n')
        weighted_fscore += f1_score(Y_test[i],Y_pred[i], \
        average='weighted')

    print("Overall average f1 score of all categories are: {}".\
    format(weighted_fscore/(i+1)))



def save_model(model, model_filepath):
    """Save trained model to sepcified file path
    """
    pickle.dump(model, open(model_filepath,'wb'))



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
