# Content Analysis and Classification of Yelp Reviews using Machine Learning

## Motivation

*Why is Natural Langauage Processing important?*  
Approximately 80% of "business-relevant information originates in unstructured form, primarily text" ([breakthroughanalysis.com](https://breakthroughanalysis.com/2008/08/01/unstructured-data-and-the-80-percent-rule/)). Obviously, if some company wants to utilize all this information, then they must be able to take this unstructured free text and turn it into something meaningful and actionable. Natural language processing (NLP) does exactly this!  

Social media is a burgeoning field built on the premise of human-to-human interaction (mainly through free text) on the internet. In this field, the ability to wrangle unstructured can provide key insights about specific users or businesses. These insights can be used to optimize marketing campaigning, recommender systems and user experience with the site or app.  

<sup> **SOURCES**  
<sup> https://breakthroughanalysis.com/2008/08/01/unstructured-data-and-the-80-percent-rule/</sup>
</sup>  

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
NlpTopicAnalysis is designed to take a pandas DataFrame of free text and create Textacy corpus of Spacy documents. Using Spacy, NlpTopicAnalysis makes it is easy to remove stop words and run tokenization, lemmatization and vectorizing operations to prepare NLP data for analysis.

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
#### Baseline:
These [GradientBoostingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) models were trained on 75,000 randomly chosen TF-IDF vectors of restaurant reviews from Yelp. These 5 models each use the same randomly chosen reviews to predict a different target/label.


  | name   |accuracy score      | target/label name |
  | ------------- |:-------------:| -----:|
  | sentiment_model |  0.71992  |  sentiment  |
  | rating_model |  0.47544  |  starsrev  |
  | usefulness_model |  0.62584  |  usefulness  |
  | price_model |  0.60988  |  RestaurantsPriceRange2  |
  | target_model |  0.27964  |  target*  |  
<sup>*Target* is a combination of rating and price range</sup>

These models will be used as a baseline to which future models will be compared.
#### GridSearch:
A [grid search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for each target/label was run on four different classification models:  
* [Gradient Boosted Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Support Vector Machine -- SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
* [Naive Bayes](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)

usefulness
(GradientBoostingClassifier(criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='deviance', max_depth=3,
                 max_features='sqrt', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=500,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False),
   {'learning_rate': 0.1, 'max_features': 'sqrt', 'n_estimators': 500},
   0.61293333333333333),

   target
   ((SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False),
   {'C': 10, 'kernel': 'linear', 'shrinking': True},
   0.28620000000000001),

   sentiment
   ((SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False),
   {'C': 10, 'kernel': 'linear', 'shrinking': True},
   0.73260000000000003),


  rating
  ((SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False),
   {'C': 1, 'kernel': 'linear', 'shrinking': True},
   0.48039999999999999)


   price
   ((RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
              max_depth=None, max_features='sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
              oob_score=False, random_state=None, verbose=0,
              warm_start=False),
  {'max_features': 'sqrt', 'n_estimators': 1000},
  0.58733333333333337),




#### Seq2Seq:

## Conclusion & Future Directions:
