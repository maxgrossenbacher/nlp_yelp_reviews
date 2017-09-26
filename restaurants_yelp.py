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

def unpack(df, column, fillna=None):
    upacked = None
    if fillna is None:
        unpacked = pd.concat([df, pd.DataFrame((d for idx, d in df[column].items()))], axis=1)
    else:
        unpacked = pd.concat([df, pd.DataFrame((d for idx, d in df[column].items())).fillna(fillna)], axis=1)
    return unpacked

def get_category(df, keywords, category='categories', b_ids='business_id', textcol='text', labels=['stars','RestaurantsPriceRange2','business_id']):
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
    scategories=[cat for cat in df[category]]
    ind = set(ind for ind, cat in enumerate(scategories) for word in cat if word in keywords)
    resturants_df = df.iloc[list(ind)]
    return resturants_df


if __name__ == '__main__':
    data_reviews = load_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/yelp_reviews.pkl")
    data_business = load_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/yelp_business.pkl")

    data_business_unpacked = unpack(data_business, 'attributes',-1)
    merged_df = data_reviews.merge(data_business_unpacked, on='business_id', how='left', suffixes=['rev', 'bus'], sort=False, indicator=True)
    keywords = ['Restaurants']
    restaurant_df = get_category(df=merged_df,keywords=keywords)
    restaurant_df.to_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/merged_rest_df.pkl")
