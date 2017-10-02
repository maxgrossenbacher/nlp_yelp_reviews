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

    '''
    Sampling for usefulness model
    '''
    # print('sampling...')
    # df.sample(frac=1)
    # df_notuseful = df[df['usefulness'] == 'not_useful'] #random sample of user reviews
    # not_useful_sample = df_notuseful.sample(n=200000)
    # df_useful = df[df['usefulness'] == 'useful']
    # useful_sample = df_notuseful.sample(n=200000)
    # veryuseful_sample = df[df['usefulness'] == 'very_useful']
    # usefulness_df = pd.concat([not_useful_sample, useful_sample, veryuseful_sample])
    # usefulness_df.to_pickle('models/usefulness_df.pkl')


    '''
    Sampling for sentiment model
    '''
    # print('sampling...')
    # df.sample(frac=1)
    # negative_df = df[df['sentiment'] == 'negative'] #random sample of user reviews
    # negative_sample = negative_df.sample(n=200000)
    # positive_df = df[df['sentiment'] == 'positive']
    # positive_sample = positive_df.sample(n=200000)
    # neutral_df = df[df['sentiment'] == 'neutral']
    # neutral_sample = neutral_df.sample(n=200000)
    # sentiment_df = pd.concat([negative_sample, positive_sample, neutral_sample])
    # sentiment_df.to_pickle('models/sentiment_df.pkl')

    '''
    Sampling for rating model
    '''
    print('sampling...')
    df.sample(frac=1)
    df_1 = df[df['starsrev'] == 1] #random sample of user reviews
    sample_1 = df_1.sample(n=100000)
    df_2 = df[df['starsrev'] == 2]
    sample_2 = df_2.sample(n=100000)
    df_3 = df[df['starsrev'] == 3]
    sample_3 = df_3.sample(n=100000)
    df_5 = df[df['starsrev'] == 5]
    sample_5 = df_5.sample(n=100000)
    df_4 = df[df['starsrev'] == 4]
    sample_4 = df_4.sample(n=100000)
    rating_df = pd.concat([sample_1, sample_2, sample_3, sample_4, sample_5])
    rating_df.to_pickle('models/rating_df.pkl')

    '''
    NLP prep for model training
    '''
    nlp = NlpTopicAnalysis(rating_df, textcol='text')
    print('processing...')
    nlp.process_text('../pkl_data', filename='rating_corpus')
    # nlp = NlpTopicAnalysis()
    # nlp.load_corpus('../pkl_data', filename='rating_corpus')
    print('vectorizing...')
    # tfidf = nlp.vectorize(weighting='tfidf')
    nlp.word2vec()
    doc_vectors=nlp.doc_vectors
    np.savez('doc_vectors_rating', doc_vectors)
    with open('rating_vectorizer.pkl', 'wb') as v:
        pickle.dump(nlp.vectorizer, v)
    # df2 = pd.read_pickle("models/df_optimization_models.pkl")

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
    training usefulness_model
    '''
    # gd_n_useful = GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt', n_estimators=500)
    # classifer(gd_n_useful, tfidf.toarray(), usefulness_df['usefulness'], name='usefulness_model_new_opt_gdc.pkl')
    '''
    sentiment model optimized
    '''
    # svc_sent_opt = SVC(C=10, kernel='linear', shrinking=True)
    # classifer(svc_sent_opt, tfidf.toarray(), sentiment_df['sentiment'], name='sentiment_model_new_opt_svc.pkl')
    '''
    rating model optimized
    '''
    gd_rating = GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt', n_estimators=500)
    classifer(gd_rating, doc_vectors, rating_df['starsrev'], name='rating_model_opt_svc.pkl')
    print('Done.')
    # svc_rating = SVC(C=1,kernel='linear',shrinking=True)
    # classifer(svc_rating, doc_vectors, rating_df['starsrev'], name='rating_model_word2vec_svc.pkl')
    # print('Done.')
