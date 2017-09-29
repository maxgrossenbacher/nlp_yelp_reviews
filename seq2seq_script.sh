#!/bin/bash
TEXT_DIR=${HOME}/text
mkdir -p ${TEXT_DIR}
# cat ../nlp_yelp_reviews/txt_files/*.txt > ${TEXT_DIR}/train_text.txt
for f in ../nlp_yelp_reviews/txt_files/*.txt; do (cat "${f}"; echo 笑) >> ${TEXT_DIR}/train_text.txt; done
wc -l < ${TEXT_DIR}/train_text.txt
# cat ../nlp_yelp_reviews/txt_label_files/*.txt > ${TEXT_DIR}/train_label.txt
for f in ../nlp_yelp_reviews/txt_label_files/*.txt; do (cat "${f}"; echo 笑) >> ${TEXT_DIR}/train_label.txt; done
wc -l < ${TEXT_DIR}/train_label.txt

head -1 ${TEXT_DIR}/train_text.txt > data_test.20
head -1 ${TEXT_DIR}/train_text.txt > data_train.80
tail -n+2 ${TEXT_DIR}/train_text.txt | awk '{if( NR % 10 <= 1){ print $0 >> "${TEXT_DIR}/data.test.20"} else {print $0 >> "${TEXT_DIR}/data.train.80"}}'

head -1 ${TEXT_DIR}/train_label.txt > data_test.labels.20
head -1 ${TEXT_DIR}/train_label.txt > data_train.labels.80
tail -n+2 ${TEXT_DIR}/train_label.txt | awk '{if( NR % 10 <= 1){ print $0 >> "${TEXT_DIR}/data_test.labels.20"} else {print $0 >> "${TEXT_DIR}/data_train.labels.80"}}'

./bin/tools/generate_vocab.py \
--max_vocab_size 50000 \
< ${TEXT_DIR}/data_train.80 > \
${TEXT_DIR}/vocab_train_text.txt

./bin/tools/generate_vocab.py \
--max_vocab_size 50000 \
< ${TEXT_DIR}/data_train.labels.80 > \
${TEXT_DIR}/vocab_train_label.txt

VOCAB_SOURCE=${TEXT_DIR}/vocab_train_text.txt
VOCAB_TARGET=${TEXT_DIR}/vocab_train_label.txt
TRAIN_SOURCES=${TEXT_DIR}/data_train.80
TRAIN_TARGETS=${TEXT_DIR}/data_train.labels.80
DEV_SOURCES=${TEXT_DIR}/data_test.20
DEV_TARGETS=${TEXT_DIR}/data_test.labels.20

# DEV_TARGETS_REF=${TEXT_DIR}/train_label.txt
TRAIN_STEPS=100000

MODEL_DIR=${HOME}/models
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
# python -m bin.infer \
#   --tasks "
#     - class: DecodeText" \
#   --model_dir $MODEL_DIR \
#   --input_pipeline "
#     class: ParallelTextInputPipeline
#     params:
#       source_files:
#         - $DEV_SOURCES" \
#   > ${PRED_DIR}/predictions.txt

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

  # python -m bin.tools.generate_beam_viz  \
  # -o ${TMPDIR:-/tmp}/beam_visualizations \
  # -d ${TMPDIR:-/tmp}/beams.npz \
  # -v ${TEXT_DIR}/vocab_train_text.txt

  ./bin/tools/multi-bleu.perl ${DEV_TARGETS} < ${PRED_DIR}/predictions.txt
