import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from latent_topic_analysis import NlpTopicAnalysis

def grid_search(X, y, estimators, params_dict, cv):
    search = GridSearchCV(estimators, params_dict, n_jobs=-1, cv=cv, scoring='f1_weighted')
    search.fit(X, y)
    search_df = pd.DataFrame(search.cv_results_)
    best_est = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    best_est_summary = (best_est, best_params, best_score)
    return best_est_summary, search_df

def creating_cv(lst, name):
    params, cvs = zip(*lst)
    usefulness_cv = pd.concat(cvs)
    usefulness_cv.to_pickle('grid_cvs/' + name+'.pkl')


if __name__ == '__main__':
    df = pd.read_pickle("../pkl_data/rest_text_target_W_ids_df.pkl")

    print('sampling...')
    df.sample(frac=1)
    df2 = df.iloc[list(range(10000))]
    nlp = NlpTopicAnalysis(df2, textcol='text')

    print('processing...')
    nlp.process_text('../pkl_data', filename='corpus_grid_search')
    print('vectorizing...')
    tfidf = nlp.vectorize(weighting='tfidf')

    grad_boost = GradientBoostingClassifier()
    random_forest = RandomForestClassifier()
    svc = SVC()
    naive_bayes = MultinomialNB()


    grad_params ={'n_estimators':(500, 1000), 'learning_rate':(0.1, 1), 'max_features':('sqrt', None)}
    svc_params = {'kernel':('linear', 'rbf'), 'C':(1, 10), 'shrinking':(True, False)}
    forest_params = {'n_estimators':(500, 1000), 'max_features':('sqrt', 10)}
    bayes_params = {'alpha':(1,0.5)}
    estimators = [(grad_boost, grad_params), (random_forest, forest_params), (svc, svc_params), (naive_bayes, bayes_params)]

    print('grid searching usefulness...')
    grid_search_usefulness = [grid_search(tfidf, df2['usefulness'], est, params, cv=4) for est, params in estimators]
    creating_cv(grid_search_usefulness, 'usefulness')
    print('grid searching target...')
    grid_search_target = [grid_search(tfidf, df2['target'], est, params, cv=4) for est, params in estimators]
    creating_cv(grid_search_target, 'target')
    print('grid searching rating...')
    grid_search_rating = [grid_search(tfidf, df2['starsrev'], est, params, cv=4) for est, params in estimators]
    creating_cv(grid_search_rating, 'rating')
    print('grid searching sentiment...')
    grid_search_sentiment = [grid_search(tfidf, df2['sentiment'], est, params, cv=4) for est, params in estimators]
    creating_cv(grid_search_sentiment, 'sentiment')
    print('grid searching price...')
    grid_search_price = [grid_search(tfidf, df2['RestaurantsPriceRange2'], est, params, cv=4) for est, params in estimators]
    creating_cv(grid_search_price, 'price')
