# Method

## Dependencies
* textacy
  * conda install -c conda-forge textacy
* SpaCy
  * conda install -c conda-forge spacy
  * python -m spacy.en.download all
* pyLDAvis
  * pip install pyldavis
* seq2seq
  * git clone https://github.com/google/seq2seq.git
  * pip install -e .
* tensorflow
  * pip install tensorflow-gpu or tensorflow
  * sudo apt-get install cuda

NlpTopicAnalysis.vectorize()
Upon experimentation of many different min_df and max_df, it appears that a min_df of 10% and a max_df of 95% provides the best TF representation of Yelp reviews

NlpTopicAnalysis.topic_analysis()
The number of topics can very depending on the business and number of reviews a business has (the more reviews, the more possible topics).

## Feature engineering
- usefulness: distinguish between useful and very-useful was somewhat arbitrary. I was trying to find ranges that would create more balanced classes.
  * not_useful  --> if useful = 0
  * useful      --> if useful = (1-4)
  * very_useful --> if useful >= 5
- sentiment: grouping star rankings into more broad classes
  * positive    --> if starsrev = (4-5)
  * neutral     --> if starsrev = 3
  * negative    --> if starsrev = (1-2)

## Model 1

### GradientBoostingClassifier
Trained on 75,000 randomly chosen TF-IDF vectors of restaurant reviews from yelp

* model target_model accuracy score: 0.27964
* pickle name: target_model.pkl
* label --> target = rating + price
  * This is a combination of rating (1-5) and price range for a given restaurant (1-4) commonly seen as ($, $$, $$$, $$$$)

---------------
* model sentiment_model accuracy score: 0.71992
* pickle name: sentiment_model.pkl
* label --> sentiment

---------------
* model rating_model accuracy score: 0.47544
* pickle name: rating_model.pkl
* label --> starsrev

---------------
* model usefulness_model accuracy score: 0.62584
* pickle name: usefulness_model.pkl
* label --> usefulness

--------------
* model price_model accuracy score: 0.60988
* pickle name: price_model.pkl
* label --> RestaurantsPriceRange2
