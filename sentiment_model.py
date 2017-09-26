import numpy as np
import pandas as pd
from latent_topic_analysis import NlpTopicAnalysis
import restaurants_yelp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle


df = pd.read_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/rest_text_target_W_ids_df.pkl")

print('sampling...')
df.sample(frac=1)
df2 = df.iloc[list(range(100000))]
nlp = NlpTopicAnalysis(df2, textcol='text')

print('processing...')
nlp.process_text('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data', filename='corpus_gdc')

print('vectorizing...')
tfidf = nlp.vectorize(weighting='tfidf')
X_train, X_test, y_train, y_text = train_test_split(tfidf.toarray(), df2['target'], stratify=labels)
gd = GradientBoostingClassifier()

print('fitting...')
gd_model = gd.fit(X_train)
print('score:',gd_model.score(X_test, y_test))
print('probablitities:', gd_model.predict_proba(X_test))

print('pickling...')
with open('gd_model.pkl', 'w') as f:
    pickle.dump(gd_model, f)

print('Done.')
