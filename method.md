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
