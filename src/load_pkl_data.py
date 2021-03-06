import numpy as np
import pandas as pd
import json


def load_json(filepath, pickle):
    '''
    DESC: Load a json file into a pandas DataFrame and export DataFrame as a pkl file
    NOTE: This script runs in python version 2.7
    --Input--
        filepath: path to json file
        pickle: filepath to save pkl
    ----------------------------------
    --Output--
        Save pkl file to (pickle) filepath
    '''
    data = pd.read_json(filepath, orient='columns', dtype=True, lines=True)
    data.to_pickle(pickle)
    return


if __name__ == '__main__':
    #create pickled dataset...I am using the Yelp challenge round 10 data
    print('reading review dataset')
    load_json("../dataset/review.json", \
                "../pkl_data/yelp_reviews.pkl")
    print('reading tip dataset')
    load_json('../dataset/tip.json', \
                '../pkl_data/yelp_tips.pkl')
    print('reading business dataset')
    load_json('../dataset/business.json', \
                '../pkl_data/yelp_business.pkl')
    print('reading user dataset')
    load_json('../dataset/user.json', \
                '../pkl_data/yelp_user.pkl')
    print('reading checkin dataset')
    load_json('../dataset/checkin.json', \
                '../pkl_data/yelp_checkin.pkl')
    print('reading photos dataset')
    load_json('../dataset/photos.json', \
                '../pkl_data/yelp_photos.pkl')
    print('Done.')
