import numpy as np
import pandas as pd
import textacy
import matplotlib.pyplot as plt


def load_pickle(pkl):
    '''
    DESC: Load a pickled pandas DataFrame
    --Input--
        pkl: filepath of pkl file
    ----------------------------------
    --Output--
        Returns pandas dataframe
    '''
    return pd.read_pickle(pkl)


def get_reviews_and_ratings(df, col, col2):
    '''
    DESC: Retrive values from a column in a pandas DataFrame and append values to a list
    --Input--
        df: pandas dataframe
        col: (str) column name
    ----------------------------------
    --Output--
        Returns 2 lists of items and labels in column of a pandas DataFrame
    '''
    text = []
    label = []
    for i in range(df.shape[0]):
        text.append(df[col].iloc[i])
        label.append(df[col2].iloc[i])
    return text, label


#Part 1
def process_text(lst, filepath=None, filename=None, compression=None):
    '''
    DESC: Tokenizes and processes lst of strings using textacy. If filepath: saves corpus as pickle to filepath.
    --Input--
        lst: list of strings
        filepath: (str) path to directory where textacy corpus will be saved
        filename: (str) name of pickled textacy corpus
        compression: (str) compression of metadata json ('gzip', 'bz2', 'lzma' or None)
    ----------------------------------
    --Output--
        Returns textacy corpus object, if filepath: saves textacy corpus as pickle
    '''
    corpus = textacy.Corpus('en', texts=lst)
    if filepath:
        corpus.save(filepath, filename, compression)
    return corpus


if __name__ == '__main__':
    #load pickled dfs
    print('loading reviews pkl...')
    data_reviews = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_reviews.pkl')
    # print('loading tips pkl...')
    # data_tips = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_tips.pkl')
    print('loading business pkl...')
    data_business = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_business.pkl')
    # print('loading user pkl...')
    # data_user = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_user.pkl')
    # print('loading checkin pkl...')
    # data_checkin = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_checkin.pkl')
    print('Done.')

    #EDA
    # sort_reviews_by_business = data_reviews[['business_id','text']].groupby('business_id').count().sort_values('text', ascending=False)
    # num_reviews_per_business = [['name','review_count','stars']].sort_values('review_count', ascending=False)
    # num_reviews_per_w_bus_id = data_business[['name', 'business_id' ,'review_count','stars']].sort_values('review_count', ascending=False)

    # print('finding most reviewed businesses...')
    # print(data_reviews[['business_id','text']].groupby('business_id').count().sort_values('text', ascending=False).head())
    # print('Done.')

    #business_id with most reviews 4JNXUYY8wbaaDmk3BPzlWw
    print('collecting reviews of business_id: 4JNXUYY8wbaaDmk3BPzlWw...')
    reviews_4JNXUYY8wbaaDmk3BPzlWw_df = data_reviews[(data_reviews['business_id'] == '4JNXUYY8wbaaDmk3BPzlWw')]
    print(type(reviews_4JNXUYY8wbaaDmk3BPzlWw_df))
    print('Done.')

    #get reviews as strings into a list
    print('collecting text...')
    reviews_4JNXUYY8wbaaDmk3BPzlWw, rating_of_review = get_reviews_and_ratings(reviews_4JNXUYY8wbaaDmk3BPzlWw_df, 'text', 'stars')
    print(type(reviews_4JNXUYY8wbaaDmk3BPzlWw))
    print('Done.')

    #tokenize reviews
    # print('processing text...')
    # corpus_4JNXUYY8wbaaDmk3BPzlWw = process_text(reviews_4JNXUYY8wbaaDmk3BPzlWw, \
    #                                                 filepath='/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data', \
    #                                                 filename='corpus_4JNXUYY8wbaaDmk3BPzlWw', \
    #                                                 compression=None)
    # print(type(corpus_4JNXUYY8wbaaDmk3BPzlWw))
    # print('Done.')
