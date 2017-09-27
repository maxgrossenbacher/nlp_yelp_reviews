import numpy as np
import pandas as pd
from latent_topic_analysis import NlpTopicAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle


df = pd.read_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/rest_text_target_W_ids_df.pkl")

print('sampling...')
df.sample(frac=1)
df2 = df.iloc[list(range(1000000))]
nlp = NlpTopicAnalysis(df2, textcol='text')

print('processing...')
nlp.process_text('../pkl_data', filename='corpus_gdc')

# nlp = NlpTopicAnalysis()
# nlp.load_corpus('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data', filename='corpus_gdc')

print('vectorizing...')
tfidf = nlp.vectorize(weighting='tfidf')
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), df2['target'], stratify=df2['target'])
X_train2, X_test2, y_train2, y_test2 = train_test_split(tfidf.toarray(), df2['sentiment'], stratify=df2['sentiment'])
X_train3, X_test3, y_train3, y_test3 = train_test_split(tfidf.toarray(), df2['starsrev'], stratify=df2['starsrev'])
X_train4, X_test4, y_train4, y_test4 = train_test_split(tfidf.toarray(), df2['useful'], stratify=df2['useful'])

gd = GradientBoostingClassifier()
gd2 = GradientBoostingClassifier()
gd3 = GradientBoostingClassifier()

print('fitting...')
gd_model = gd.fit(X_train, y_train)
gd2_model =gd2.fit(X_train2, y_train2)
gd3_model =gd3.fit(X_train3, y_train3)
gd4_model =gd4.fit(X_train4, y_train4)

print('scoring')
print('model accuracy score:',gd_model.score(X_test, y_test))
print('model probabilites:', gd_model.predict_proba(X_test))

print('model2 accuracy score:', gd2_model.score(X_test2, y_test2))
print('model2 probabilites:', gd2_model.predict_proba(X_test2))

print('model3 accuracy score:', gd3_model.score(X_test3, y_test3))
print('model3 probabilites:', gd3_model.predict_proba(X_test3))

print('model3 accuracy score:', gd4_model.score(X_test4, y_test4))
print('model3 probabilites:', gd4_model.predict_proba(X_test4))

print('pickling...')
with open('gd_model.pkl', 'wb') as f:
    pickle.dump(gd_model, f)

print('pickling...')
with open('gd2_model.pkl', 'wb') as f:
    pickle.dump(gd2_model, f)

print('pickling...')
with open('gd3_model.pkl', 'wb') as f:
    pickle.dump(gd3_model, f)

print('pickling...')
with open('gd4_model.pkl', 'wb') as f:
    pickle.dump(gd4_model, f)

print('Done.')
