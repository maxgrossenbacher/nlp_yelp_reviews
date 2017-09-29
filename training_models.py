import numpy as np
import pandas as pd
from latent_topic_analysis import NlpTopicAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle


def classifer(model, X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model.fit(X_train, y_train)
    print(name+'_accuracy:',model.score(X_test, y_test))
    with open(name, 'wb') as f:
        pickle.dump(model, f)
    return

if __name__ == '__main__':
    '''
    Data load for model training
    '''
    df = pd.read_pickle("../pkl_data/rest_text_target_W_ids_df.pkl")
    print('sampling...')
    df.sample(frac=1)
    df2 = df.iloc[list(range(1000000))] #random sample of user reviews
    df2.to_pickle('models/df_optimization_models.pkl')

    '''
    NLP prep for model training
    '''
    nlp = NlpTopicAnalysis(df2, textcol='text')
    print('processing...')
    nlp.process_text('../pkl_data', filename='training_optimization_data')
    # nlp = NlpTopicAnalysis()
    # nlp.load_corpus('../pkl_data', filename='training_optimized')
    print('vectorizing...')
    tfidf = nlp.vectorize(weighting='tfidf')

    df2 = pd.read_pickle("models/df_optimized_models.pkl")

    '''
    vanilla model training
    '''
    # gd = GradientBoostingClassifier()
    # classifer(gd, tfidf.toarray(), df2['target'], name='target_model')
    # classifer(gd, tfidf.toarray(), df2['sentiment'], name='sentiment_model')
    # classifer(gd, tfidf.toarray(), df2['starsrev'], name='rating_model')
    # classifer(gd, tfidf.toarray(), df2['usefulness'], name='usefulness_model')
    # classifer(gd, tfidf.toarray(), df2['RestaurantsPriceRange2'], name='price_model')

    '''
    training grid usefulness_model
    '''
    # gd_useful = GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt', n_estimators=500)
    # classifer(gd_useful, tfidf.toarray(), df2['usefulness'], name='usefulness_model')

    '''
    training grid target_model
    '''
    # svc_target = SVC(C=10, kernel='linear', shrinking=True)
    # classifer(svc_target, tfidf.toarray(), df2['target'], name='target_model')

    '''
    training grid sentiment_model
    '''
    # svc_sent = SVC(C=10, kernel='linear', shrinking=True)
    # classifer(svc_sent, tfidf.toarray(), df2['sentiment'], name='sentiment_model')

    '''
    training grid rating_model
    '''
    # svc_rating = SVC(C=1,kernel='linear',shrinking=True)
    # classifer(svc_rating, tfidf.toarray(), df2['starsrev'], name='rating_model')

    '''
    training grid price_model
    '''
    # rand_price = RandomForestClassifier(max_features='sqrt', n_estimators=1000)
    # classifer(rand_price, tfidf.toarray(), df2['RestaurantsPriceRange2'], name='price_model')

    '''
    training grid usefulness_model
    '''
    gd_n_useful = GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt', n_estimators=1000)
    classifer(gd_n_useful, tfidf.toarray(), df2['usefulness'], name='usefulness_model_opt_gdc')

    print('Done.')
