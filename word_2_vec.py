
from latent_topic_analysis import NlpTopicAnalysis
import spacy
import textacy
import numpy as np
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

def punct_space(token):
    return token.is_punct or token.is_space

def lemmatized_sentence_corpus(corpus):
    for doc in corpus:
        for sent in doc.sents:
            yield ' '.join([token.lemma_ for token in sent] if not punct_space(token))

with open('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews/unigram_sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in lemmatized_sentence_corpus(nlp.corpus):
        f.write(sentence + '\n')

unigram_sentences = LineSentence('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews/unigram_sentences.txt')

with open('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews/bigram_sentences.txt', 'w', encoding='utf-8') as f:
    for uni_sentence in unigram_sentences:
        bigram_sentence = ' '.join(bigram_model[uni_sentence])
        f.write(bigram_sentence + '\n')

bigram_sentences = LineSentence('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews/bigram_sentences.txt')

with open('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews/trigram_sentences.txt', 'w', encoding='utf-8') as f:
    for bi_sent in bigram_sentences:
        trigram_sentence = ' '.join(trigram_model[bi_sent])
        f.write(trigram_sentence + '\n')

trigram_sentences = LineSentence('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews/bigram_sentences.txt')

with open('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/nlp_yelp_reviews/trigram_review.txt', 'w', encoding='utf-8') as f:
    for doc in nlp.corpus:
        unigram_review = [token.lemma_ for token in doc if not punct_space(token)]

        bigram_review = bigram_model[unigram_review]
        trigram_review = trigram_model[bigram_review]

        trigram_review = [term for term in trigram_review if term not in spacy.en.STOPWORDS]

        trigram_review = ' '.join(trigram_review)
        f.write(trigram_review + '\n')
