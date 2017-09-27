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
    data_reviews = load_pickle("../pkl_data/yelp_reviews.pkl")
    data_business = load_pickle("../pkl_data/yelp_business.pkl")

    print('unpacking attributes...')
    data_business_unpacked = unpack(data_business, 'attributes')
    print('Done.')
    print('creating sentiment col...')
    data_reviews['sentiment'] = data_reviews['stars'].apply(lambda x: 'negative'if x <3 else ('neutral' if x==3 else 'positive'))
    data_reviews['usefulness'] = data_reviews['useful'].apply(lambda x: 'very_useful'if x >=5 else ('useful' if  x>0 else 'not_useful'))
    print('Done.')
    print('merging dfs & finding restaurants...')
    merged_df = data_reviews.merge(data_business_unpacked, on='business_id', how='left', suffixes=['rev', 'bus'], sort=False, indicator=True)
    keywords = ['Restaurants']
    restaurant_df = get_category(df=merged_df,keywords=keywords)
    restaurant_df.reset_index(inplace=True)
    print('Done.')
    print('creating rest_text_target_w_ids df...')
    rest_text_target_w_ids = restaurant_df[['business_id','review_count','text', 'starsrev', 'RestaurantsPriceRange2', 'sentiment', 'useful', 'funny', 'cool']]
    rest_text_target_w_ids.dropna(inplace=True)
    rest_text_target_w_ids['target'] = rest_text_target_w_ids['starsrev'].map(str) + '-' + rest_text_target_w_ids['RestaurantsPriceRange2'].map(str)
    print('Done.')
    print('pickling df...')
    rest_text_target_w_ids.to_pickle("../pkl_data/rest_text_target_W_ids_df.pkl")
    print('Done.')
