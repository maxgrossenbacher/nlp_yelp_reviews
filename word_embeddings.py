import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from latent_topic_analysis import NlpTopicAnalysis
import latent_topic_analysis
import csv
import os
import numpy as np


print('loading reviews pkl...')
data_reviews = latent_topic_analysis.load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_reviews.pkl')
print('loading business pkl...')
data_business = latent_topic_analysis.load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_business.pkl')
print('Done.')

#business_id with most reviews 4JNXUYY8wbaaDmk3BPzlWw
print('collecting reviews of business_id: 4JNXUYY8wbaaDmk3BPzlWw...')
reviews_4JNXUYY8wbaaDmk3BPzlWw_df = latent_topic_analysis.business_reviews(data_reviews, 'business_id', '4JNXUYY8wbaaDmk3BPzlWw')
# print(type(reviews_4JNXUYY8wbaaDmk3BPzlWw_df))
print('Done.')

nlp = NlpTopicAnalysis(reviews_4JNXUYY8wbaaDmk3BPzlWw_df, 'text', 'stars')
nlp.process_text()
nlp.word2vec()
np.savetxt('metadata.tsv', nlp.label, delimiter='\t')



sess = tf.InteractiveSession()
LOG_DIR = '/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews'
doc_embeddings = tf.Variable(nlp.doc_vectors, trainable=False, name='embedding')
tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = doc_embeddings.name
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

projector.visualize_embeddings(writer, config)
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 10000)
