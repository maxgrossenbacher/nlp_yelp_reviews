
# coding: utf-8

# # EDA Notebook

# In[1]:


import pandas as pd
import numpy as np
import retriving_reviews
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.misc import imread
get_ipython().magic('matplotlib inline')


# ### Importing as the Yelp DataSets

# In[2]:


data_reviews = retriving_reviews.load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_reviews.pkl')
data_business = retriving_reviews.load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_business.pkl')
# data_tips = retriving_reviews.load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_tips.pkl')
# data_user = retriving_reviews.load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_user.pkl')
#data_photos = retriving_reviews.load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_photos.pkl')
# data_checkin = retriving_reviews.load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_checkin.pkl')


# ## REVIEW DATASET

# In[3]:


print(data_reviews.describe())
print(data_reviews.info())
data_reviews.head()


# In[4]:


text = data_reviews.text #these are the reviews that I will be processing
text.head()


# In[5]:


data_reviews.text.iloc[2] #some reviews not in english


# In[6]:


data_reviews['sentiment'] = data_reviews['stars'].apply(lambda x: 'negative'if x <3 else ('neutral' if x==3 else 'positive'))
print(data_reviews['text'][(data_reviews['sentiment'] == 'negative')].head())
data_reviews.head()


# In[7]:


data_reviews.sentiment.value_counts()


# In[8]:


pd.DataFrame.hist(data=data_reviews, column='useful', bins=10)


# In[9]:


data_reviews['usefulness'] = data_reviews['useful'].apply(lambda x: 'very_useful'if x >=5 else ('useful' if  x>0 else 'not_useful'))
data_reviews.head()


# In[10]:


data_reviews['usefulness'].value_counts()


# In[11]:


print(data_reviews['stars'].value_counts())
data_reviews['stars'].mean() # mean review rating


# ## BUSINESS DATASET

# In[12]:


print(data_business.describe())
print(data_business.info())
data_business.head()


# ### Important summary stats
# #### Business Rating
# * mean rating=3.647154
# * std=0.977640
# * min rating=1
# * max rating=5
#     * 25%=3.0
#     * 50%=3.5
#     * 75%=4.5
#         * break up sentiment anaylsis --> below 50% negative, above 50% positive
# 
# #### Number of Reviews per Business
# * mean rating=30.238159
# * std=96.486631   
# * min=3.000000 
# * max=6979.000000
#     * 25%=4.000000   
#     * 50%=9.000000   
#     * 75%=23.000000  

# In[4]:


data_business.city.value_counts() #unique locations and counts
data_business.city.count()


# In[5]:


data_business['stars'].unique()


# In[6]:


data_business.attributes.iloc[4]


# ## Getting Only Restaurants

# In[3]:


cats = data_business.categories
print(cats.shape)
cats.head(10)


# Need to figure out which key terms to split by in categories possibly Restaurants

# In[4]:


scategories=[cat for cat in cats]
keywords = ['Restaurants']
ind = set(ind for ind, cat in enumerate(scategories) for word in cat if word in keywords)
len(ind)


# In[5]:


restaurant_ids = data_business['business_id'].iloc[list(ind)]
print(len(restaurant_ids))
restaurant_ids.head()


# There are 51,613 Business with the category Restaurant

# In[6]:


data_rest = data_business[(data_business['business_id'].isin(restaurant_ids))]
data_rest.head()


# In[ ]:


attributes = data_rest['attributes']
print(attributes.head())


# In[ ]:


for a in data_rest['attributes']:
    if 'RestaurantsPriceRange2' in a:
        data_rest['price'] = (a['RestaurantsPriceRange2'])
    else:
        data_rest['price'] = 0
data_rest.head()


# In[ ]:


rest_att = pd.DataFrame(attributes)
rest_att.head()


# ## Restaurant_Review DataFrame

# In[7]:


restaurant_review_df = data_reviews[data_reviews['business_id'].isin(restaurant_ids)]
print(restaurant_review_df.describe())
print(restaurant_review_df.info())
restaurant_review_df.head()


# #### Getting reviews and labels

# In[8]:


text, review_rating = [], []
for i in range(restaurant_review_df.shape[0]):
    text.append(restaurant_review_df['text'].iloc[i])
    review_rating.append(restaurant_review_df['stars'].iloc[i])
print(len(text), len(review_rating))
text[1], review_rating[1]


# 2,927,731 reviews of 51,613 Restaurants! Wow thats a lot!!!

# In[9]:


print(restaurant_review_df['stars'].unique())
fig4 = plt.figure(figsize=(10,10))
ax5 = fig4.add_subplot(111)
sns.distplot(restaurant_review_df['stars'], hist=True, kde=False, ax=ax5, color='magenta')
plt.xlabel('Rating', fontsize=15)
plt.ylabel('Number of Reviews', fontsize=15)
plt.title('Restaurant Review Ratings', fontsize=15)
plt.show()


# ### Finding Top 50 most reviewed Restaurants

# In[10]:


restaurant_business = data_business[(data_business['business_id'].isin(restaurant_ids))]
restaurant_business_most_reviewed = restaurant_business.sort_values('review_count', ascending=False)
restaurant_business_most_reviewed['business_id'].head()


# ### Resturant Review Rating
# 
# * mean rating=3.702
# * std=1.34
# * min rating=1
# * max rating=5
#     * 25%=3.0
#     * 50%=4.0
#     * 75%=5.0
#  

# In[11]:


top_50_rated = [restaurant_business_most_reviewed['business_id'].iloc[i] for i in range(50)]
print(top_50_rated)
rated_restaurant_reviews_50 = restaurant_review_df[(restaurant_review_df['business_id'].isin(top_50_rated))]
print(rated_restaurant_reviews_50.head())


# business ids of 50 most rated restaurants

# ### Barplot of Distribution of Reviews

# In[14]:


plt_reviews = rated_restaurant_reviews_50[['business_id','stars']]
dummies = pd.get_dummies(plt_reviews, columns=['stars'])
dummies = dummies.groupby('business_id').sum().sort_values(by='stars_5')
print(dummies.head())

ax = dummies.plot.barh(stacked=True, figsize=(20,20), title='Rating Distribution of the Top 50 most reviewed Resturants', fontsize=15)
fig = ax.get_figure()
fig.savefig('/Users/gmgtex/Desktop/nlp_yelp_reviews/50_most_rated.png')
plt.show()


# ### Restaurant Locations

# In[25]:


img = imread("/Users/gmgtex/Desktop/Map_Plots/World_map.png")
fig = plt.figure(figsize=(15,15))
restaurant_business['state_count'] = restaurant_business.groupby('state')['state'].transform(pd.Series.value_counts)
print(restaurant_business.head())
ax = sns.lmplot(x ='longitude',y = 'latitude', data = restaurant_business,hue='state_count', palette='Reds',                fit_reg = False, size=15, scatter_kws = {'alpha':.01}, legend=False)
plt.title("Where Yelp Businesses are Located",fontsize = 30)
plt.xlabel("Longitude",fontsize = 20)
plt.ylabel("Latitude",fontsize = 20)
plt.imshow(img,extent = [-180,180,-90,90])
plt.show()


# Most restaurants in this dataset seem to be located in the United States

# ### Average Rating of Resturants

# In[26]:


fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax.hist(restaurant_business['stars'], bins=9, color='orange')
ax.set_xlabel('Rating', fontsize=15)
ax.set_ylabel('Number of Restaurants', fontsize=15)

ax.set_title('Average Restaurant Ratings', fontsize=15)
ax2.hist(data_business['stars'], bins=9, color='purple')
ax2.set_xlabel('Rating', fontsize=15)
ax2.set_ylabel('Number of Businesses', fontsize=15)
ax2.set_title('Average Business Ratings', fontsize=15)
ax.axvline(3.702, color='purple',linestyle='--')
ax2.axvline(3.647, color='orange', linestyle='--')
plt.tight_layout()
plt.show()


# In[27]:


fig2 = plt.figure(figsize=(10,10))
ax3 = fig2.add_subplot(121)
ax4 = fig2.add_subplot(122)
sns.violinplot(y="stars", data=restaurant_business, ax=ax3, bins=9)
sns.violinplot(y='stars', data=data_business, ax=ax4, bins=9)


# ## TIPS DATASET

# In[23]:


print(data_tips.describe())
print(data_tips.info())
data_tips.head()


# In[29]:


tips = data_tips.text 
tips.iloc[1]


# Tips could be looked at as shorter review like objects?

# ## USER DATASET

# In[30]:


print(data_user.describe())
print(data_user.info())
data_user.head()


# ## Shower Thoughts on User Dataset
# * looks like users get a rating also --> higher rating could mean more reliable
# * Does elite mean trustworthy or just someone that does a lot or reviews?
#     * FROM YELP WEBSITE:
#         * "What does it take to be Elite? Elite-worthiness is based on well-written reviews, high quality photos, an active voting and complimenting record, and a history of playing well with others."
#         * You have to apply to be Elite
# * useful count maybe an indicator of real review vs. fake/spam review.

# In[ ]:


print(data_checkin.describe())
print(data_checkin.info())
data_checkin.head()


# ## PHOTOS DATASET

# In[3]:


data_photos.head()


# In[4]:


data_photos.info()


# In[6]:


data_photos.describe()


# In[7]:


data_photos.label.value_counts()


# In[ ]:


sort_reviews_by_business = data_reviews[['business_id','text']].groupby('business_id').count().sort_values('text', ascending=False)
num_reviews_per_business = [['name','review_count','stars']].sort_values('review_count', ascending=False)
num_reviews_per_w_bus_id = data_business[['name', 'business_id' ,'review_count','stars']].sort_values('review_count', ascending=False)


# In[ ]:




