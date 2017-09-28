# Content Analysis and Classification of Yelp Reviews using Machine Learning

## Motivation

Yelp is one of the most popular user review sites in the world.



## Overview

#### Part 1:
Create natural language processing pipeline

### Part 2:
Finding latent topics/keywords in yelp reviews

#### Part 3:
Use machine learning to create models to predict rating and usefulness of yelp reviews



## The Data:

I used the Yelp data challenge 10 dataset made publicly available by Yelp. This data set comes with a reviews.json, business.json, checkin.json, users.json and photos.json. For this project, I focused on the reviews and businesses data.

I was able to isolate over ~3 M reviews of over 51,000 businesses containing the category keyword restaurant.

#### Feature engineering
- **usefulness**: distinguish between useful and very-useful was somewhat arbitrary. I was trying to find ranges that would create more balanced classes.

|not_useful|useful|very_useful|
|----------|:------:|--------:|
| useful = 0 |  0 < useful < 5 | useful >= 5|

- **sentiment**: grouping star rankings into more broad classes

|negative|neutral|positive|
|----------|:------:|--------:|
| starsrev < 3 |  starsrev = 3 | starsrev > 3|

- **RestaurantsPriceRange2***: price rating of a restaurant (1-4) commonly seen as ($, $$, $$$, $$$$)




## Part 1:
#### Building NLP Pipeline:
Using the most reviewed business in the yelp dataset, I was able to build my NlpTopicAnalysis class. This class is designed to take a pandas DataFrame and create Textacy corpus of Spacy documents.  
NlpTopicAnalysis makes it is easy to remove stop words and run tokenization, lemmatization and vectorizing operations to prepare nlp data for analysis.

## Part 2:
#### Keyword Detection of reviews:
Once NLP data has been processed. NlpTopicAnalysis allows latent topic modeling using NMF, LDA (Latent Dirichlet Allocation) or LSA. For my purposes, I chose to model the yelp reviews using LDA.  
<sup>A quick note: when modeling using LDA best results are achieved using term-freq (TF) matrix.</sup>  
Below is a example termite plot of latent topics.  
![alt text](termiteplot_lda4JNXUYY8wbaaDmk3BPzlWw.png)  
<sup>* The bigger the circle, the more important the term is to the topic. The colored topics show the 5 most important topics</sup>  
Additionally, NlpTopicAnalysis can create a interactive pyLDAvis plot of these latent topics.

## Part 3:
#### Machine learning classification of reviews:
Using Sklearn's GradientBoostingClassifier, these models were trained on 75,000 randomly chosen TF-IDF vectors of restaurant reviews from Yelp. These 5 models each use the same randomly chosen reviews to predict a different target/label.


  | name   |accuracy score      | target/label name |
  | ------------- |:-------------:| -----:|
  | sentiment_model |  0.71992  |  sentiment  |
  | rating_model |  0.47544  |  starsrev  |
  | usefulness_model |  0.62584  |  usefulness  |
  | price_model |  0.60988  |  RestaurantsPriceRange2  |
  | target_model |  0.27964  |  target*  |  
<sup>**Target* is a combination of rating (1-5) and price range for a given restaurant (1-4) commonly seen as ($, $$, $$$, $$$$)</sup>

These models will be used as a baseline to which future models will be compared.
