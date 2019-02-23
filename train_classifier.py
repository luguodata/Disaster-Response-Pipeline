import sys


def load_data(database_filepath):

    # load data from database
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('ETL_processed_data', engine)

    return df

# take a look of df
df.head()


def tokenize(text):

    # Normailization -- lower case + remove puntuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())

    # tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    words = [word for word in tokens if word not in stopwords.words("english")]

    # stemmization
    text_lems = [WordNetLemmatizer().lemmatize(lem).strip() for lem in words]

    return text_lems


def build_model():
    """
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [20, 50],
        'clf__estimator__max_depth': [3, 6],
        'clf__estimator__min_samples_split': [2,4]
    }

    cv = GridSearchCV(pipeline, param_grid= parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    """
    weighted_fscore = 0
    for i in np.arange(0,36,1):
        print("Target:{}".format(category_names[i]))
        print("\n")
        print(classification_report(Y_test[i],Y_pred[i]))
        print('\n')
        print('\n')
        weighted_fscore += f1_score(Y_test[i],Y_pred[i], average='weighted')

    print("Overall average f1 score of all categories are: {}".format(weighted_fscore/(i+1)))


def save_model(model, model_filepath):
    pass


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
