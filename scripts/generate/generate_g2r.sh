#!/bin/zsh
set -eux

BIENCODER_MODEL_FILE=$1
RESULTS_PATH=${2:-./results/faiss_results.jsonl}
RESPONSE_FILE_STEM="bst_data_level_g2r_responses"
FIXED_CANDIDATES_PATH=./datasets/emnlp_2021_g2r_dataset/${RESPONSE_FILE_STEM}.txt
ENCS_FILE_PATH="${BIENCODER_MODEL_FILE}.${RESPONSE_FILE_STEM}.encs"
FAISS_INDEX_FILE="${BIENCODER_MODEL_FILE}.${RESPONSE_FILE_STEM}.encs.faissidx"


python3 maybe_build_biencoder_encs.py \
  -m transformer/biencoder \
  -mf $BIENCODER_MODEL_FILE \
  --eval-candidates fixed \
  --encode-candidate-vecs-batchsize 4096 \
  --fixed-candidates-path $FIXED_CANDIDATES_PATH \
  --encs-file-path $ENCS_FILE_PATH

python3 maybe_build_faiss_index.py \
  --input-encs-path $ENCS_FILE_PATH \
  --index-save-path $FAISS_INDEX_FILE \

python3 generate_result_onlybi.py \
  -m transformer/biencoder \
  -mf $BIENCODER_MODEL_FILE \
  --biencoder-model-file $BIENCODER_MODEL_FILE \
  --eval-candidates fixed \
  --encode-candidate-vecs-batchsize 4096 \
  --fixed-candidates-path $FIXED_CANDIDATES_PATH \
  --faiss-index-path $FAISS_INDEX_FILE \
  --input-path ./assets/bst_test_dialogues_200.json \
  --result-save-path $RESULTS_PATH \
  --search-topk 4096
