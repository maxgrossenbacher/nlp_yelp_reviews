import numpy as np
import pandas as pd
from latent_topic_analysis import NlpTopicAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import pickle
from sklearn.externals import joblib



def classifer(model, X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(name+'_accuracy:',model.score(X_test, y_test))
    joblib.dump(model, name)
    return model, preds, y_test

if __name__ == '__main__':
    '''
    Data load for model training
    '''
    print('load df...')
    df = pd.read_pickle("../pkl_data/rest_text_target_W_ids_df.pkl")

    '''
    Sampling for usefulness model
    '''
    print('sampling...')
    df.sample(frac=1)
    df_notuseful = df[df['usefulness'] == 'not_useful'] #random sample of user reviews
    not_useful_sample = df_notuseful.sample(n=100000)
    df_useful = df[df['usefulness'] == 'useful']
    useful_sample = df_useful.sample(n=100000)
    veryuseful_df = df[df['usefulness'] == 'very_useful']
    veryuseful_sample = veryuseful_df.sample(n=100000)
    usefulness_df = pd.concat([not_useful_sample, useful_sample, veryuseful_sample])
    usefulness_df.to_pickle('models/usefulness_df.pkl')


    '''
    Sampling for sentiment model
    '''
    # print('sampling...')
    # df.sample(frac=1)
    # negative_df = df[df['sentiment'] == 'negative'] #random sample of user reviews
    # negative_sample = negative_df.sample(n=100000)
    # positive_df = df[df['sentiment'] == 'positive']
    # positive_sample = positive_df.sample(n=100000)
    # neutral_df = df[df['sentiment'] == 'neutral']
    # neutral_sample = neutral_df.sample(n=100000)
    # sentiment_df = pd.concat([negative_sample, positive_sample, neutral_sample])
    # sentiment_df.to_pickle('models/sentiment_df.pkl')

    '''
    Sampling for rating model
    '''
    # print('sampling...')
    # df.sample(frac=1)
    # df_1 = df[df['starsrev'] == 1] #random sample of user reviews
    # sample_1 = df_1.sample(n=50000)
    # df_2 = df[df['starsrev'] == 2]
    # sample_2 = df_2.sample(n=50000)
    # df_3 = df[df['starsrev'] == 3]
    # sample_3 = df_3.sample(n=50000)
    # df_5 = df[df['starsrev'] == 5]
    # sample_5 = df_5.sample(n=50000)
    # df_4 = df[df['starsrev'] == 4]
    # sample_4 = df_4.sample(n=50000)
    # rating_df = pd.concat([sample_1, sample_2, sample_3, sample_4, sample_5])
    # rating_df.to_pickle('models/rating_df.pkl')

    '''
    NLP prep for model training
    '''
    nlp = NlpTopicAnalysis(usefulness_df, textcol='text')
    print('processing...')
    nlp.process_text('../pkl_data', filename='usefulness_corpus')
    # nlp = NlpTopicAnalysis()
    # nlp.load_corpus('../pkl_data', filename='usefulness_corpus')
    print('vectorizing...')
    # tfidf = nlp.vectorize(weighting='tfidf')
    nlp.word2vec()
    doc_vectors=nlp.doc_vectors
    np.save('doc_vectors_usefulness', doc_vectors)
    print('loaded doc vectors...')
    doc_vectors = np.load('doc_vectors_usefulness.npy')
    # with open('usefulness_vectorizer.pkl', 'wb') as v:
        # pickle.dump(nlp.vectorizer, v)
    # usefulness_df = pd.read_pickle("models/usefulness_df.pkl")


    '''
    training usefulness_model
    '''
    rf_n_useful = RandomForestClassifier(max_features='sqrt', n_estimators=1000)
    print('training model...')
    model, preds, y_test = classifer(rf_n_useful, doc_vectors, usefulness_df['usefulness'], name='usefulness_model_gbc_word2vec.pkl')
    print('f1_score:', f1_score(y_test, preds, average='weighted'))
    # print('training model...')
    # model_2, preds_2, y_test_2 = classifer(rf_n_useful, tfidf.toarray(), usefulness_df['usefulness'], name='usefulness_model_gbc_tfidf.pkl')
    # print('f1_score:', f1_score(y_test_2, preds_2, average='weighted'))
    # print('Done.')
    '''
    sentiment model optimized
    '''
    # print('training model...')
    # gb_sentiment = GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt', n_estimators=500)
    # model3, preds3, y_test3 = classifer(gb_sentiment, doc_vectors, sentiment_df['sentiment'], name='sentiment_model_gbc_word2vec.pkl')
    # print('f1_score:', f1_score(y_test3, preds3, average='weighted'))
    # model4, preds4, y_test4 = classifer(gd_sentiment, tfidf.toarray(), sentiment_df['sentiment'], name='sentiment_model_gbc_tfidf.pkl')
    # print('f1_score:', f1_score(y_test4, preds4, average='weighted'))
    '''
    rating model optimized
    '''
    # gd_rating = GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt', n_estimators=500)
    # model5, preds5, y_test5 = classifer(gd_rating, doc_vectors, rating_df['starsrev'], name='rating_model_gbc_word2vec.pkl')
    # print('f1_score:', f1_score(y_test5, preds5, average='weighted'))
    # model6, preds6, y_test6 = classifer(gd_rating, tfidf.toarray(), rating_df['starsrev'], name='rating_model_gbc_tfidf.pkl')
    # print('f1_score:', f1_score(y_test6, preds6, average='weighted'))
    # print('Done.')
    #
