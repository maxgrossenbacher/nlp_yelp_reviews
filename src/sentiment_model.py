import numpy as np
import pandas as pd
from latent_topic_analysis import NlpTopicAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle


def classifer(model, X, y, name):
    '''
    DESC: This function splits data and fits a model.
    --Input--
        model: sklearn model
        X: feature matrix
        y: targets
        name: filpath to save fitted model
    ----------------------------------
    --Output--
        Returns accuracy score, proabilities of certain class
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    proba = model.predict_proba(X_train)
    with open(name, 'wb') as f:
        pickle.dump(model, f)
    print('model' + name +  'accuracy score:',score)
    print('model' + name + 'probabilites:', proba)
    return score, proba

if __name__ == '__main__':
    df = pd.read_pickle("../pkl_data/rest_text_target_W_ids_df.pkl")

    print('sampling...')
    df.sample(frac=1)
    df2 = df.iloc[list(range(100000))]
    nlp = NlpTopicAnalysis(df2, textcol='text')

    print('processing...')
    nlp.process_text('../pkl_data', filename='corpus_gdc')

    # nlp = NlpTopicAnalysis()
    # nlp.load_corpus('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data', filename='corpus_gdc')

    print('vectorizing...')
    tfidf = nlp.vectorize(weighting='tfidf')

    gd = GradientBoostingClassifier()

    score1, probabilities1 = classifer(gd, tfidf.toarray(), df2['target'], name='target_model')
    score2, probabilities2 = classifer(gd, tfidf.toarray(), df2['sentiment'], name='sentiment_model')
    score3, probabilities3 = classifer(gd, tfidf.toarray(), df2['starsrev'], name='rating_model')
    # score4, probabilities4 = classifer(gd, tfidf.toarray(), df2['useful'], name='useful_model')
    score5, probabilities5 = classifer(gd, tfidf.toarray(), df2['usefulness'], name='usefulness_model')
    score6, probabilities6 = classifer(gd, tfidf.toarray(), df2['RestaurantsPriceRange2'], name='price_model')
    print('Done.')
