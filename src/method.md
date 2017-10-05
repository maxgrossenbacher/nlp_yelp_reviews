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


usefulness
(GradientBoostingClassifier(criterion='friedman_mse', init=None,
                 learning_rate=0.1, loss='deviance', max_depth=3,
                 max_features='sqrt', max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None,
                 min_samples_leaf=1, min_samples_split=2,
                 min_weight_fraction_leaf=0.0, n_estimators=500,
                 presort='auto', random_state=None, subsample=1.0, verbose=0,
                 warm_start=False),
   {'learning_rate': 0.1, 'max_features': 'sqrt', 'n_estimators': 500},
   0.61293333333333333),

   target
   ((SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False),
   {'C': 10, 'kernel': 'linear', 'shrinking': True},
   0.28620000000000001),

   sentiment
   ((SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False),
   {'C': 10, 'kernel': 'linear', 'shrinking': True},
   0.73260000000000003),


  rating
  ((SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False),
   {'C': 1, 'kernel': 'linear', 'shrinking': True},
   0.48039999999999999)


   price
   ((RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
              max_depth=None, max_features='sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
              oob_score=False, random_state=None, verbose=0,
              warm_start=False),
  {'max_features': 'sqrt', 'n_estimators': 1000},
  0.58733333333333337),


AttentionSeq2Seq:
  attention.class: seq2seq.decoders.attention.AttentionLayerDot
  attention.params: {num_units: 128}
  bridge.class: seq2seq.models.bridges.ZeroBridge
  bridge.params: {}
  decoder.class: seq2seq.decoders.AttentionDecoder
  decoder.params:
    rnn_cell:
      cell_class: GRUCell
      cell_params: {num_units: 128}
      dropout_input_keep_prob: 0.8
      dropout_output_keep_prob: 1.0
      num_layers: 1
  embedding.dim: 128
  embedding.init_scale: 0.04
  embedding.share: false
  encoder.class: seq2seq.encoders.BidirectionalRNNEncoder
  encoder.params:
    rnn_cell:
      cell_class: GRUCell
      cell_params: {num_units: 128}
      dropout_input_keep_prob: 0.8
      dropout_output_keep_prob: 1.0
      num_layers: 1
  inference.beam_search.beam_width: 0
  inference.beam_search.choose_successors_fn: choose_top_k
  inference.beam_search.length_penalty_weight: 0.0
  optimizer.clip_embed_gradients: 0.1
  optimizer.clip_gradients: 5.0
  optimizer.learning_rate: 0.0001
  optimizer.lr_decay_rate: 0.99
  optimizer.lr_decay_steps: 100
  optimizer.lr_decay_type: ''
  optimizer.lr_min_learning_rate: 1.0e-12
  optimizer.lr_staircase: false
  optimizer.lr_start_decay_at: 0
  optimizer.lr_stop_decay_at: 2147483647
  optimizer.name: Adam
  optimizer.params: {epsilon: 8.0e-07}
  optimizer.sync_replicas: 0
  optimizer.sync_replicas_to_aggregate: 0
  source.max_seq_len: 50
  source.reverse: false
  target.max_seq_len: 50
  vocab_source: /Users/gmgtex/Desktop/Galvanize/Immersive/capstone/seq2seq/text/vocab_train_text.txt
  vocab_target: /Users/gmgtex/Desktop/Galvanize/Immersive/capstone/seq2seq/text/vocab_train_label.txt


  BidirectionalRNNEncoder:
  init_scale: 0.04
  rnn_cell:
    cell_class: GRUCell
    cell_params: {num_units: 128}
    dropout_input_keep_prob: 0.8
    dropout_output_keep_prob: 1.0
    num_layers: 1
    residual_combiner: add
    residual_connections: false
    residual_dense: false

INFO:tensorflow:Creating AttentionLayerDot in mode=eval
INFO:tensorflow:
AttentionLayerDot: {num_units: 128}

INFO:tensorflow:Creating AttentionDecoder in mode=eval
INFO:tensorflow:
AttentionDecoder:
  init_scale: 0.04
  max_decode_length: 100
  rnn_cell:
    cell_class: GRUCell
    cell_params: {num_units: 128}
    dropout_input_keep_prob: 0.8
    dropout_output_keep_prob: 1.0
    num_layers: 1
    residual_combiner: add
    residual_connections: false
    residual_dense: false
