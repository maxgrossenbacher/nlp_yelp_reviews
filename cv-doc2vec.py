from latent_topic_analysis import NlpTopicAnalysis
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.externals import joblib

df = pd.read_pickle("../pkl_data/rest_text_target_W_ids_df.pkl")

print('sampling...')
df.sample(frac=1)
df2 = df.iloc[list(range(10000))]
nlp = NlpTopicAnalysis(df2, textcol='text')
nlp.process_text('../pkl_data', filename='corpus_cvdoc2vec')
nlp.word2vec()
doc2vec = nlp.doc_vectors

usefulness_model = joblib.load('models/usefulness_model_rf_word2vec.pkl')
sentiment_model = joblib.load('models/sentiment_model_gbc_word2vec.pkl')
rating_model = joblib.load('models/rating_model_gbc_word2vec.pkl')

scores_use = cross_validate(usefulness_model, doc2vec, df2['usefulness'], scoring='f1_weighted',cv=4, return_train_score=True)
scores_sent = cross_validate(sentiment_model, doc2vec, df2['sentiment'], scoring='f1_weighted',cv=4, return_train_score=True)
scores_rating = cross_validate(rating_model, doc2vec, df2['starsrev'], scoring='f1_weighted',cv=4, return_train_score=True)
scores_price = cross_validate(rating_model, doc2vec, df2['RestaurantsPriceRange2'], scoring='f1_weighted',cv=4, return_train_score=True)
scores_target = cross_validate(rating_model, doc2vec, df2['target'], scoring='f1_weighted',cv=4, return_train_score=True)

def score(dic):
    test = dic['test_score'].mean()
    train = dic['train_score'].mean()
    return test, train

print(score(scores_use))
print(score(scores_sent))
print(score(scores_rating))
print(score(scores_price))
print(score(scores_target))
