#!/bin/bash

cat ../nlp_yelp_reviews/txt_files/*.txt >> train_text.txt
cat ../nlp_yelp_reviews/txt_label_files/*.txt >> train_label.txt

TEXT_DIR=${HOME}/text

./bin/tools/generate_vocab.py \
--max_vocab_size 50000 \
< ../nlp_yelp_reviews/txt_label_files/train_label.txt > \
${TEXT_DIR}/vocab_train_label.txt
--max_vocab_size 50000 \
./bin/tools/generate_vocab.py \
< ../nlp_yelp_reviews/txt_files/train_text.txt > \
${TEXT_DIR}/vocab_train_text.txt


VOCAB_SOURCE=${TEXT_DIR}/vocab_train_label.txt
VOCAB_TARGET=${TEXT_DIR}/vocab_train_text.txt
TRAIN_SOURCES=${HOME}/nlp_yelp_reviews/txt_label_files/train_label.txt
TRAIN_TARGETS=${HOME}/nlp_yelp_reviews/txt_files/train_text.txt
DEV_SOURCES=${HOME}/nlp_yelp_reviews/txt_label_files/train_label.txt
DEV_TARGETS=${HOME}/nlp_yelp_reviews/txt_files/train_text.txt

DEV_TARGETS_REF=${HOME}/nlp_yelp_reviews/txt_files/train_text.txt
TRAIN_STEPS=1000

MODEL_DIR=${TMPDIR:-/tmp}/nmt_tutorial
PRED_DIR=${MODEL_DIR}/pred


mkdir -p $MODEL_DIR
python -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR

mkdir -p ${PRED_DIR}
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${PRED_DIR}/predictions.txt

python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${PRED_DIR}/predictions.txt

  ./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt
