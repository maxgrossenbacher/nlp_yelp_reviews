#!/bin/bash
VOCAB_SOURCE=${HOME}/nmt_data/toy_reverse/train/vocab.sources.txt
VOCAB_TARGET=${HOME}/nmt_data/toy_reverse/train/vocab.targets.txt
TRAIN_SOURCES=${HOME}/nmt_data/toy_reverse/train/sources.txt
TRAIN_TARGETS=${HOME}/nmt_data/toy_reverse/train/targets.txt
DEV_SOURCES=${HOME}/nmt_data/toy_reverse/dev/sources.txt
DEV_TARGETS=${HOME}/nmt_data/toy_reverse/dev/targets.txt

DEV_TARGETS_REF=${HOME}/nmt_data/toy_reverse/dev/targets.txt
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
