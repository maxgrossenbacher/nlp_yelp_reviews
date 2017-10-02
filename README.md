# Content Analysis and Classification of Yelp Reviews using Machine Learning

## Motivation

*Why is Natural Langauage Processing important?*  
Approximately 80% of "business-relevant information originates in unstructured form, primarily text" ([breakthroughanalysis.com](https://breakthroughanalysis.com/2008/08/01/unstructured-data-and-the-80-percent-rule/)). Obviously, if some company wants to utilize all this information, then they must be able to take this unstructured free text and turn it into something meaningful and actionable. Natural language processing (NLP) does exactly this!  

Social media is a burgeoning field built on the premise of human-to-human interaction (mainly through free text) on the internet. In this field, the ability to wrangle unstructured can provide key insights about specific users or businesses. These insights can be used to optimize marketing campaigning, recommender systems and user experience with the site or app.  


## Overview
This project is built around 3 main questions and explores the power of natural language processing to process and analyze text.
#### Question 1:
Can I create build a scaleable and reusable natural language processing pipeline?
#### Question 2:
Can I find latent topics/keywords for business on Yelp based solely on user reviews of that business?
#### Question 3:
Can I use machine learning to create models to predict rating, usefulness and sentiment of yelp review?



## The Data:

[Yelp's Challenge Dataset](https://www.yelp.com/dataset/challenge) provides access to millions of user reviews. This data set comes with a reviews.json, business.json, checkin.json, users.json and photos.json. For this project, I focused on the reviews and businesses data.

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

- **RestaurantsPriceRange2**: price rating of a restaurant (1-4) commonly seen as ($, $$, $$$, $$$$)


## Part 1:
#### Building NLP Pipeline:
[NlpTopicAnalysis](https://github.com/maxgrossenbacher/nlp_yelp_reviews/blob/master/latent_topic_analysis.py) is designed to take a pandas DataFrame of free text and create Textacy corpus of Spacy documents. Using Spacy, NlpTopicAnalysis makes it is easy to remove stop words and run tokenization, lemmatization and vectorizing operations to prepare NLP data for analysis.

## Part 2:
#### Keyword Detection of reviews:
Once NLP data has been processed. NlpTopicAnalysis allows latent topic modeling using NMF, LDA (Latent Dirichlet Allocation) or LSA. For my purposes, I chose to model the yelp reviews using LDA.  
<sup>A quick note: when modeling using LDA best results are achieved using term-freq (TF) matrix.</sup>  
Below is a example termite plot of latent topics.  
![alt text](https://github.com/maxgrossenbacher/nlp_yelp_reviews/blob/master/termiteplot_lda2iTsRqUsPGRH1li1WVRvKQ.png)
<sup>* The bigger the circle, the more important the term is to the topic. The colored topics show the 5 most important topics</sup>  
Additionally, NlpTopicAnalysis can create a interactive pyLDAvis plot of these latent topics.

## Part 3:
#### Machine learning classification of reviews:
#### Baseline:
These GradientBoostingClassifier models were trained on 75,000 randomly chosen TF-IDF vectors of restaurant reviews from Yelp. These 5 models each use the same randomly chosen reviews to predict a different target/label.


  | name   |accuracy | target/label name |
  | ------------- |:-------------:| -----:|
  | usefulness_model |  62.6%  |  usefulness  |
  | target_model |  28.0%  |  target*  |
  | sentiment_model |  72.0%  |  sentiment  |
  | rating_model |  47.5%  |  starsrev (rating)  |
  | price_model |  61.0%  |  RestaurantsPriceRange2 (price)  |


<sup>*Target* is a combination of rating and price range</sup>

These models will be used as a baseline to which future models will be compared.
#### GridSearch:
A [grid search](https://github.com/maxgrossenbacher/nlp_yelp_reviews/blob/master/grid_search.py) for each target/label was run on four different classification models:  
* [Gradient Boosted Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Support Vector Machine -- SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
* [Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)  

Models were trained on 10,000 random user reviews with a 4-Fold cross validation
#### Grid Search Best Models & Parameters
| model   |       parameters        | accuracy | target/label name |
| ------------- |:---------------------------------:|:-------------:|-----:|
| GradientBoostingClassifier | {'learning_rate': 0.1, 'max_features': 'sqrt', 'n_estimators': 500} | 62.3% | usefulness |   
| SVC | {'C': 10, 'kernel': 'linear', 'shrinking': True} | 29.1% | target |   
| SVC | {'C': 10, 'kernel': 'linear', 'shrinking': True} | 73.3% | sentiment |
| SVC | {'C': 1, 'kernel': 'linear', 'shrinking': True} | 48.0% | rating |
| RandomForestClassifier | {'max_features': 'sqrt', 'n_estimators': 1000} | 58.7% | price |

<sup> Full Grid Search CVs can be found in grid_cvs </sup>

59.4% sentiment
61.1% usefulness


#### Seq2Seq:

## Web App:
[Yelp Review Scorer]() will process a Yelp-type review and output a sentiment, rating and usefulness score. Scores are based on the best models and parameters obtained after a grid search. Have fun!

## Conclusion & Future Directions:
