import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



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


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('etl_processed_data', engine)

# load model
model = joblib.load("model/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # add1: related unrelated message proportion compare
    related_cnt = df.groupby('related').count()['message']
    related_label = related_cnt.index.tolist()
    related_label = ['N','Y']


    # add2: message categories sorted counts within related messages
    categry_cnts = df.ix[:,5:].sum().sort_values(ascending = False)
    categry_names = categry_cnts.index.tolist()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=related_label,
                    y=related_cnt
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Related Messages',
                'titlefont': {
                'size': 20
                     },
                'yaxis': {
                    'title': "Count",
                    'titlefont':{
                    'size': 16
                    }
                },
                'xaxis': {
                    'title': "Disaster Related",
                    'titlefont':{
                    'size': 16
                    }
                }
            }
        },

        {
            'data': [
                Bar(
                    x=categry_names,
                    y=categry_cnts

                )
            ],

            'layout': {
                'title': 'Category Distribution of Disaster Messages',
                'titlefont': {
                'size': 20
                },
                'xaxis': {
                    'title': "Disaster Categories",
                    'titlefont':{
                    'size': 16
                    },
                    'tickangle': -20,
                    'tickfont': {
                    'size': 10
                    }
                },

                 'yaxis': {
                    'title': "Count",
                    'titlefont':{
                    'size': 16
                    }
                },
                'autosize' : False,
                'width': 1200,
                'height': 600
            }
        }





    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    print(query)
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

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
