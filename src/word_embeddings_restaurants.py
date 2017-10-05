import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from latent_topic_analysis import NlpTopicAnalysis
import latent_topic_analysis
import restaurants_yelp as ry
import csv
import os
import numpy as np


data_reviews = ry.load_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/yelp_reviews.pkl")
data_business = ry.load_pickle("/Users/gmgtex/Desktop/Galvanize/immersive/capstone/pkl_data/yelp_business.pkl")

keywords = ['Restaurants']
restaurant_reviews, restaurant_review_labels = ry.get_category(data_business,data_reviews, keywords)

nlp = NlpTopicAnalysis(text=restaurant_reviews, label=restaurant_review_labels)
nlp.process_text()
nlp.word2vec()
np.savetxt('restaurants_metadata.tsv', nlp.label, delimiter='\t')



sess = tf.InteractiveSession()
LOG_DIR = '/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews/word_embed_restaurant'
doc_embeddings = tf.Variable(nlp.doc_vectors, trainable=False, name='embedding')
tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = doc_embeddings.name
embedding.metadata_path = os.path.join(LOG_DIR, 'restaurants_metadata.tsv')

projector.visualize_embeddings(writer, config)
saver.save(sess, os.path.join(LOG_DIR, "model_restaurants.ckpt"), 10000)
