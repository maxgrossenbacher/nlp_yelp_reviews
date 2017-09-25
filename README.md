# Content Analysis Of Yelp Reviews using Seq2Seq Neural Network

## Motivation

Yelp is one of the most popular user review sites in the world.



## Overview

#### Part 1:
Use Natural language processing techniques to find latent topics in yelp reviews.

#### Part 2:
Use word2vec modeling to cluster yelp reviews between different Yelp businesses.

#### Part 2:
Use tensorflow seq2seq neural network to create a summary or typical review for restaurant based on reviews that a restaurant has received on yelp.


### The Data:

I used the Yelp data challenge 10 dataset made publicly available by Yelp. This data set comes with a reviews.json, business.json, checkin.json, users.json and photos.json. For my project, I focused on the reviews and businesses data.

I was able to isolate over 2,000,000 reviews of ~51,000 businesses containing the category keyword restaurant.


## Part 1:
Proof of concept:
Using the most reviewed business in the yelp dataset, I was able to build my NlpTopicAnalysis class. This class is designed to take a pandas DataFrame and create Textacy corpus of Spacy documents. Then, you can pass this corpus through a tf vectorizer in order to prepare the data for Latent Dirichlet Allocation in order to find latent topics in the user created reviews.
Below is a termite plot of latent topics. The bigger the circle, the more important the term is to the topic.
![alt text](termite_plot_4JNXUYY8wbaaDmk3BPzlWw_lda.png)
Additionally, I was able to create a pyLDAvis interactive plot of these latent topics.

## Part 2:
Using GloVe word2vec available through the Spacy library. I was able to create 300 feature representations of each Yelp review. Then using word_embeddings through tensorflow, I was able to create a 3d interative plot in which each review/point is labeled by rating and business_id. Tensorflow allows the user to choose if they prefer to use PCA or t-SNE dimensionality reduction of the 300-feature space into 2 or 3 dimensions.

## Part 3:
