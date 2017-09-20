import numpy as np
import pandas as pd
import textacy
import spacy
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

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

    def _get_reviews_and_ratings(self):
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
        self._get_reviews_and_ratings()
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
            Returns uncompressed textacy corpus object
        '''
        self.corpus = textacy.Corpus.load(filepath, filename, compression)
        return

    # Part4 Vectorizing textacy corpus
    def vectorize(self):
        '''
        DESC: Creates tfidf matrix of textacy corpus.
        --Output--
            Returns tfidf matrix, list of terms, and vectorizer object used to create tfidf matrix
        '''
        terms_list = []
        for doc in self.corpus:
            terms_list.append(doc.to_terms_list(ngrams=1, named_entities=True, \
                                                normalize='lemma', as_strings=True, \
                                                filter_stops=True, filter_punct=True ))
        self.vectorizer = textacy.Vectorizer(weighting='tfidf', normalize=True, \
                                            smooth_idf=True, min_df=10, max_df=0.9)
        self.tfidf = self.vectorizer.fit_transform(terms_list)
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
            n_clusters: number of cluster to model
        ----------------------------------
        --Output--
            Returns centroids for n_clusters and labels for each tfidf vector
        '''
        knn = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, \
                        tol=0.0001, precompute_distances='auto', verbose=0, \
                        random_state=None, copy_x=True, n_jobs=-1, algorithm='auto')
        knn.fit(self.pca_mat)
        centroids = knn.cluster_centers_
        k_labels = knn.labels_
        if self.pca_mat.shape[1] == 2:
            plt.scatter(self.pca_mat[:,0], self.pca_mat[:,1], c=k_labels.astype(np.float), alpha=0.3)
        return centroids, k_labels

    def sentiment_analysis(self, labels):
        nb = MultinomialNB(alpha=1.0)
        X_train, X_test, y_train, y_test = train_test_split(self.tfidf, labels, test_size=0.33)
        nb.fit(X_train, y_train)
        proabilities = nb.predict_proba(X_test)
        predictions = nb.predict(X_test)
        return proabilities, nb.score(X_test, y_test)

    #Part 2
    def topic_analysis(self, n_topics=5, model_type='nmf', n_terms=25, n_highlighted_topics=1, save=False):
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
        topic_w_weights = {}
        keys = []
        self.model = textacy.TopicModel(model_type, n_topics=n_topics, init='random', alpha=0.2, maxiter=5000, l1_ratio=.9)
        self.model.fit(self.tfidf)
        topic_matrix = self.model.transform(self.tfidf)
        for topic_idx, top_terms in self.model.top_topic_terms(self.vectorizer.id_to_term, topics=range(n_topics), weights=False):
            print('Topic {}: {}' .format(topic_idx, top_terms))
        for topic, weight in enumerate(self.model.topic_weights(topic_matrix)):
            topic_w_weights[topic] = weight
            print('Topic {} has weight: {}' .format(topic, weight))
        sort_keys = sorted(topic_w_weights.values())[::-1]
        highlight = [topic_w_weights[i] for i in sort_keys[:n_highlighted_topics]]
        self.model.termite_plot(self.tfidf, self.vectorizer.id_to_term, topics=-1,  n_terms=n_terms, highlight_topics=highlight)
        if save:
            plt.savefig()
        return topic_matrix, topic_w_weights


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

    #EDA
    # sort_reviews_by_business = data_reviews[['business_id','text']].groupby('business_id').count().sort_values('text', ascending=False)
    # num_reviews_per_business = [['name','review_count','stars']].sort_values('review_count', ascending=False)
    # num_reviews_per_w_bus_id = data_business[['name', 'business_id' ,'review_count','stars']].sort_values('review_count', ascending=False)

    # print('finding most reviewed businesses...')
    # print(data_reviews[['business_id','text']].groupby('business_id').count().sort_values('text', ascending=False).head())
    # print('Done.')

    #business_id with most reviews 4JNXUYY8wbaaDmk3BPzlWw
    print('collecting reviews of business_id: 4JNXUYY8wbaaDmk3BPzlWw...')
    reviews_4JNXUYY8wbaaDmk3BPzlWw_df = data_reviews[(data_reviews['business_id'] == '4JNXUYY8wbaaDmk3BPzlWw')]
    # print(type(reviews_4JNXUYY8wbaaDmk3BPzlWw_df))
    print('Done.')
    #
    # #get reviews as strings into a list
    # print('collecting text...')
    # reviews_4JNXUYY8wbaaDmk3BPzlWw, rating_of_review = get_reviews_and_ratings(reviews_4JNXUYY8wbaaDmk3BPzlWw_df, 'text', 'stars')
    # print(type(reviews_4JNXUYY8wbaaDmk3BPzlWw))
    # print('Done.')
    #
    # #tokenize reviews
    # print('tokenizing text and compressing corpus...')
    # corpus_4JNXUYY8wbaaDmk3BPzlWw = process_text(reviews_4JNXUYY8wbaaDmk3BPzlWw, \
    #                                                 filepath='/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data', \
    #                                                 filename='corpus_4JNXUYY8wbaaDmk3BPzlWw', \
    #                                                 compression=None)
    # print(type(corpus_4JNXUYY8wbaaDmk3BPzlWw))
    # print('Done.')


    #load corpus of textacy docs
    # print('loading textacy corpus from compressed file...')
    # corpus_4JNXUYY8wbaaDmk3BPzlWw = load_corpus(filepath='/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data', \
    #                                             filename='corpus_4JNXUYY8wbaaDmk3BPzlWw', \
    #                                             compression=None)

    # print('k-means clustering and pca...')
    # doc_term_matrix, vectorizer = vectorize(corpus_4JNXUYY8wbaaDmk3BPzlWw)
    # tfidf_2d = pca(doc_term_matrix.toarray())
    # centroids, labels = k_means(tfidf_2d, 5)
    # print('Done.')
    #
    #
    # print('Naive Bayes rating anaylsis...')
    # probablites, accuracy = sentiment_analysis(doc_term_matrix.toarray(), rating_of_review)
    # print('Done.')
    #create models
    # print('creating nmf model...')
    # topic_matrix_nmf, topic_w_weights_nmf = topic_analysis(doc_term_matrix, \
    #                                                         vectorizer, \
    #                                                         n_topics=12, \
    #                                                         model_type='nmf', \
    #                                                         n_terms=50, \
    #                                                         n_highlighted_topics=5)
    # plt.tight_layout()
    # plt.savefig('termite_plot_nmf_4JNXUYY8wbaaDmk3BPzlWw')
    # plt.show()
    # print('Done.')
    #
    #
    # print('creating lda model...')
    # topic_matrix_lda, topic_w_weights_lda = topic_analysis(doc_term_matrix, \
    #                                                         vectorizer, \
    #                                                         n_topics=12, \
    #                                                         model_type='lda', \
    #                                                         n_terms=50, \
    #                                                         n_highlighted_topics=5)
    # plt.tight_layout()
    # plt.savefig('termite_plot_lda_4JNXUYY8wbaaDmk3BPzlWw')
    # plt.show()
    # print('Done.')
    #
    #
    # print('creating lsa model...')
    # topic_matrix_lsa, topic_w_weights_lsa = topic_analysis(doc_term_matrix, \
    #                                                         vectorizer, \
    #                                                         n_topics=12, \
    #                                                         model_type='lsa', \
    #                                                         n_terms=50, \
    #                                                         n_highlighted_topics=5)
    # plt.tight_layout()
    # plt.savefig('termite_plot_lsa_4JNXUYY8wbaaDmk3BPzlWw')
    # plt.show()
    # print('Done.')


# nlp.process_text(filepath='/Users/gmgtex/Desktop/Galvanize/Immersive/capstone/pkl_data',filename='corpus_4JNXUYY8wbaaDmk3BPzlWw', compression=None)
