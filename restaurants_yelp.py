import pandas as pd
import numpy as np

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

def get_category(keywords):
    '''
    DESC: Grab businesses with certain keywords in the categories list of
            yelp_business dataframe and return their reviews.
    --Input--
        keywords: list of strings containing keywords to search for in
                    categories column of data_business df
    ----------------------------------
    --Output--
        Returns list of reviews and list of ratings for the review
    '''
    keywords = keywords
    cats = data_business.categories
    scategories=[cat for cat in cats]
    ind = set(ind for ind, cat in enumerate(scategories) for word in cat if word in keywords)
    restaurant_ids = data_business['business_id'].iloc[list(ind)]
    restaurant_review_df = data_reviews[data_reviews['business_id'].isin(restaurant_ids)]
    restaurant_business = data_business[(data_business['business_id'].isin(restaurant_ids))]
    text, review_rating = [], []
    for i in range(restaurant_review_df.shape[0]):
        text.append(restaurant_review_df['text'].iloc[i])
        review_rating.append(restaurant_review_df['stars'].iloc[i])
    print(len(text), len(review_rating))
    print(text[1], review_rating[1])
    return text, review_rating


if __name__ == '__main__':
    data_reviews = load_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/yelp_reviews.pkl")
    data_business = load_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/yelp_business.pkl")

    keywords = ['Restaurants']
    restaurant_reviews, restaurant_review_labels = get_category(keywords)
