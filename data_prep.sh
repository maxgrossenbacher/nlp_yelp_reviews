#!/bin/bash
TEXT_DIR=${HOME}/Desktop/Galvanize/Immersive/capstone/seq2seq/text
SUBWORDS=${HOME}/Desktop/Galvanize/Immersive/capstone/subword-nmt
mkdir -p ${TEXT_DIR}
# cat ../nlp_yelp_reviews/txt_files/*.txt > ${TEXT_DIR}/train_text.txt
for f in ../nlp_yelp_reviews/txt_files/*.txt; do (cat "${f}"; echo 笑) >> ${TEXT_DIR}/train_text.txt; done
# cat ../nlp_yelp_reviews/txt_label_files/*.txt > ${TEXT_DIR}/train_label.txt
for f in ../nlp_yelp_reviews/txt_label_files/*.txt; do (cat "${f}"; echo 笑) >> ${TEXT_DIR}/train_label.txt; done



head -1 ${TEXT_DIR}/train_text.txt > data_test.20.txt
head -1 ${TEXT_DIR}/train_text.txt > data_train.80.txt
tail -n+2 ${TEXT_DIR}/train_text.txt | awk '{if( NR % 10 <= 1){ print $0 >> "data_test.20.txt"} else {print $0 >> "data_train.80.txt"}}'



head -1 ${TEXT_DIR}/train_label.txt > data_test.labels.20.txt
head -1 ${TEXT_DIR}/train_label.txt > data_train.labels.80.txt
tail -n+2 ${TEXT_DIR}/train_label.txt | awk '{if( NR % 1000 <= 1){ print $0 >> "data_test.labels.20.txt"} else {print $0 >> "data_train.labels.80.txt"}}'


./bin/tools/generate_vocab.py \
--max_vocab_size 700000 \
< data_train.80.txt > \
${TEXT_DIR}/vocab_train_text.txt

./bin/tools/generate_vocab.py \
--delimiter "" \
--max_vocab_size 50000 \
< data_train.labels.80.txt > \
${TEXT_DIR}/vocab_train_label.txt


${SUBWORDS}/learn_bpe.py -s 10000 < data_train.80.txt > codes.bpe
${SUBWORDS}/apply_bpe.py -c codes.bpe < data_train.80.txt > source_train.bpe
${SUBWORDS}/apply_bpe.py -c codes.bpe < data_test.20.txt > source_test.bpe

${SUBWORDS}/learn_bpe.py -s 10000 < data_train.labels.80.txt > codes_labels.bpe
${SUBWORDS}/apply_bpe.py -c codes_labels.bpe < data_train.labels.80.txt > target_train.bpe
${SUBWORDS}/apply_bpe.py -c codes_labels.bpe < data_test.labels.20.txt > target_test.bpe
