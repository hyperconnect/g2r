#!/bin/zsh
MODEL_PATH=$1

python3 generate_result_parlai.py \
  -m transformer/biencoder \
  -mf $MODEL_PATH \
  --eval-candidates fixed \
  --encode-candidate-vecs-batchsize 4096 \
  --rank-top-k 1 \
  --input-path ./assets/bst_test_dialogues_200.json \
  --fixed-candidates-path ./datasets/emnlp_2021_g2r_dataset/bst_original_responses.txt \
  --result-save-path ./results/biencoder_only_results.jsonl \
