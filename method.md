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

##Model 1

### GradientBoostingClassifier
Trained on 500,000 randomly chosen TF-IDF vectors of restaurant reviews from yelp
* target = rating + price
  * This is a combination of rating (1-5) and price range for a given restaurant (1-4) commonly seen as ($, $$, $$$, $$$$)
* model accuracy score: 0.286992
* pickle name: gd_model.pkl
probabilities:
[[ 0.03495921  0.13957568  0.01405628 ...,  0.14261414  0.01703212, 0.00276594]
 [ 0.01557103  0.02011498  0.01157794 ...,  0.20121069  0.04308675, 0.00831571]
 [ 0.03646812  0.12580839  0.0092277  ...,  0.07689576  0.0072433, 0.00147423]
 ...,
 [ 0.07331464  0.20274931  0.02198528 ...,  0.04637214  0.0096605, 0.00358445]
 [ 0.0120366   0.0176673   0.00239195 ...,  0.22568942  0.01694309, 0.00252535]
 [ 0.00752734  0.01267515  0.00152356 ...,  0.21451662  0.01503242, 0.00222793]]


* target = sentiment
  * ratings of 4 or 5 --> positive
  * ratings of 3 --> neutral
  * ratings of 1 or 2 --> negative
* model2 accuracy score: 0.71464799999999995
* pickle name: gd2_model.pkl
probabilities model 2:
[[ 0.09607983  0.20593298  0.69798719]
 [ 0.0190285   0.0237046   0.9572669 ]
 [ 0.35552966  0.3001465   0.34432383]
 ...,
 [ 0.36171558  0.17060382  0.4676806 ]
 [ 0.2977544   0.13619129  0.56605432]
 [ 0.41116803  0.30834134  0.28049063]]
