
from latent_topic_analysis import NlpTopicAnalysis
import textacy
import spacy
import sense2vec
import numpy as np
nlp = NlpTopicAnalysis()
nlp.load_corpus(filepath='/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data', \
                filename='corpus_4JNXUYY8wbaaDmk3BPzlWw', \
                compression=None)
nlp.vectorize()
# reviews = []
# for doc in nlp.corpus:
#     reviews.append(doc.spacy_doc.text)
# model = Word2Vec(reviews, size=100, window=5, min_count=5, workers=4)
# model.build_vocab(reviews)
# model.train(reviews, total_examples=model.corpus_count, epochs=10)
# model.wv.vocabwo
# model.save('word2vec_model')
# model = Word2Vec.load('word2vec_model')

doc_vects = []
toks_vects = []
toks = []
for ind, doc in enumerate(nlp.corpus):
    print('going through doc {}...'.format(ind))
    doc_vects.append(doc.spacy_doc.vector)
    for token in doc.spacy_doc:
        toks_vects.append(token.vector)
        toks.append(token)
print('creating arrays')
print(len(doc_vects))
doc_v = np.array(doc_vects)
print(len(toks_vects))
tok_v = np.array(toks_vects)
print('Done.')
