import numpy as np
import pandas as pd
import textacy
import spacy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#Part 1 retriving reviews from pkl df
def load_pickle(pkl):
    '''
    DESC: Load a pickled pandas DataFrame
    --Input--
        pkl: filepath of pkl file
    ----------------------------------
    --Output--
        Returns pandas dataframe
    '''
    return pd.read_pickle(pkl)



class NlpTopicAnalysis(object):
    """
    DESC: Takes a pandas DataFrame and allow nlp processing using textacy
    --Input--
        df: pandas dataframe
        textcol: (str) name of column in pandas DataFrame where text are located
        labelcol: (str) name of column in pandas DataFrame where labels are located
    """
    def __init__(self, df, textcol, labelcol=None):
        self.df = df
        self.textcol = textcol
        self.labelcol = labelcol
        self.vectorizer = None
        self.corpus = None
        self.model = None
        self.text = []
        self.label = []
        self.pca_mat = None
        self.tfidf = None
        self.topic_matrix = None
        self.latent_topics_top_terms = {}
        self.terms_list = []
        self.topic_w_weights = {}

    def _get_reviews_and_label(self):
        '''
        DESC: Retrive values from a column in a pandas DataFrame and append values to a list
        --Input--
            df: pandas dataframe
            col: (str) column name
        ----------------------------------
        --Output--
            Returns 2 lists of items and labels in column of a pandas DataFrame
        '''
        for i in range(self.df.shape[0]):
            self.text.append(self.df[self.textcol].iloc[i])
            if self.labelcol:
                self.label.append(self.df[self.labelcol].iloc[i])
        return


    #Part 2 Saving textacy corpus as compressed file
    def process_text(self, filepath=None, filename=None, compression=None):
        '''
        DESC: Tokenizes and processes pandas DataFrame using textacy. If filepath: saves corpus as pickle to filepath.
        --Input--
            filepath: (str) path to directory where textacy corpus will be saved
            filename: (str) name of pickled textacy corpus
            compression: (str) compression of metadata json ('gzip', 'bz2', 'lzma' or None)
        ----------------------------------
        --Output--
            Returns textacy corpus object, if filepath: saves textacy corpus as pickle
        '''
        self._get_reviews_and_label()
        self.corpus = textacy.Corpus('en', texts=self.text)
        if filepath:
            self.corpus.save(filepath, filename, compression)
            print('Saved textacy corpus to filepath.')
        return


    #Part 3 loading textacy corpus from compressed file
    def load_corpus(self, filepath, filename, compression=None):
        '''
        DESC: Loads pickled corpus of textacy/spacy docs
        --Input--
            filepath: (str) path to directory where textacy corpus is located
            filename: (str) name of pickled textacy corpus
            compression: (str) compression of pickled textacy corpus (gzip, 'bz2', 'lzma' or None)
        ----------------------------------
        --Output--
            Returns textacy corpus object
        '''
        self.corpus = textacy.Corpus.load(filepath, filename, compression)
        return

    # Part4 Vectorizing textacy corpus
    def vectorize(self, min_df=100, max_df=0.95, max_n_terms=100000):
        '''
        DESC: Creates tfidf matrix of textacy corpus.
        --Output--
            Returns creates tfidf matrix of textacy corpus.
        '''
        for doc in self.corpus:
            self.terms_list.append(list(doc.to_terms_list(n_grams=1, named_entities=True, \
                                                normalize='lemma', as_strings=True, \
                                                filter_stops=True, filter_punct=True )))
        self.vectorizer = textacy.Vectorizer(weighting='tfidf', normalize=True, \
                                            smooth_idf=True, min_df=min_df, max_df=max_df, max_n_terms=max_n_terms)
        self.tfidf = self.vectorizer.fit_transform(self.terms_list)
        return

    # Principle component analysis for k means graph
    def pca(self, n_components):
        p = PCA(n_components=n_components, copy=True, whiten=False, svd_solver='auto', \
                    tol=0.0, iterated_power='auto', random_state=None)
        self.pca_mat = p.fit_transform(self.tfidf.toarray())
        return


    def k_means(self, n_clusters):
        '''
        DESC: K-nearest neighbors modeling of tfidf matrix.
        --Input--
            n_clusters: number of clusters to model
        ----------------------------------
        --Output--
            Returns centroids for n_clusters and labels for each tfidf document vector
        '''
        knn = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=100, \
                        tol=0.0001, precompute_distances='auto', verbose=0, \
                        random_state=None, copy_x=True, n_jobs=-1, algorithm='auto')
        knn.fit(self.pca_mat)
        centroids = knn.cluster_centers_
        k_labels = knn.labels_
        if self.pca_mat.shape[1] == 2:
            plt.scatter(self.pca_mat[:,0], self.pca_mat[:,1], c=k_labels.astype(np.float), alpha=0.3)
        return centroids, k_labels

    #Part 2
    def topic_analysis(self, n_topics=5, model_type='nmf', n_terms=25, n_highlighted_topics=1, plot=False, save=False, kwargs=None):
        '''
        DESC: Latent topic modeling of tfidf matrix. Generates termite plot of latent topics.
        for corpus on n topics
        --Input--
            n_topics: (int) number of latent topics
            model_type: (str) 'nmf','lsa','lda'
            n_terms: (int) number of key terms ploted in termite plot (y-axis)
            n_highlighted_topics: (int) number of highlighted key topics sorted by importance, max highlighted topics is 6
            save = filename to save plot
        ----------------------------------
        --Output--
            Returns topic_matrix of num_docs X n_topics dimensions, topic weights/importance for each topic, and termite plot of key terms to latent topics
        '''
        if n_highlighted_topics > 6:
            print('Value Error: n_highlighted_topics must be =< 5')
            return
        highlighting = {}
        self.model = textacy.TopicModel(model_type, n_topics=n_topics, kwargs=kwargs)
        self.model.fit(self.tfidf)
        self.topic_matrix = self.model.transform(self.tfidf)
        for topic_idx, top_terms in self.model.top_topic_terms(self.vectorizer.id_to_term, topics=range(n_topics), weights=False):
            self.latent_topics_top_terms[topic_idx] = top_terms
            print('Topic {}: {}' .format(topic_idx, top_terms))
        for topic, weight in enumerate(self.model.topic_weights(self.topic_matrix)):
            self.topic_w_weights[topic] = weight
            highlighting[weight] = topic
            # print('Topic {} has weight: {}' .format(topic, weight))
        if plot:
            sort_weights = sorted(highlighting.keys())[::-1]
            highlight = [highlighting[i] for i in sort_weights[:n_highlighted_topics]]
            self.model.termite_plot(self.tfidf, self.vectorizer.id_to_term, topics=-1,  n_terms=n_terms, highlight_topics=highlight)
            if save:
                plt.savefig(save)
            plt.tight_layout()
            plt.show()
        return


if __name__ == '__main__':
    print('loaded NlpTopicAnalysis')
    #load pickled dfs
    print('loading reviews pkl...')
    data_reviews = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_reviews.pkl')
    # # print('loading tips pkl...')
    # # data_tips = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_tips.pkl')
    # # print('loading business pkl...')
    # # data_business = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_business.pkl')
    # # print('loading user pkl...')
    # # data_user = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_user.pkl')
    # # print('loading checkin pkl...')
    # # data_checkin = load_pickle('/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data/yelp_checkin.pkl')
    print('Done.')

    #business_id with most reviews 4JNXUYY8wbaaDmk3BPzlWw
    print('collecting reviews of business_id: 4JNXUYY8wbaaDmk3BPzlWw...')
    reviews_4JNXUYY8wbaaDmk3BPzlWw_df = data_reviews[(data_reviews['business_id'] == '4JNXUYY8wbaaDmk3BPzlWw')]
    # print(type(reviews_4JNXUYY8wbaaDmk3BPzlWw_df))
    print('Done.')

    nlp = NlpTopicAnalysis(reviews_4JNXUYY8wbaaDmk3BPzlWw_df, 'text', 'stars')
    nlp.process_text()
    nlp.vectorize()

    print(nlp.terms_list)
    print(nlp.vectorizer.vocabulary)
    nlp.topic_analysis(model_type='lda',n_topics=20, n_terms=50, \
                                n_highlighted_topics=5, \
                                kwargs={'learning_method':'batch', 'max_iter':25}, plot=True)
    nlp.topic_analysis(model_type='nmf',n_topics=20, n_terms=50, n_highlighted_topics=5, plot=True, kwargs={'init':'nndsvd', 'max_iter':25, 'solver':'mu'})
