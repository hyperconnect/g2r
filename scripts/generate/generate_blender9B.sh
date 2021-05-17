#!/bin/zsh
python3 generate_result_parlai.py \
  -m transformer/generator \
  -mf zoo:blender/blender_9B/model \
  --inference beam \
  --beam-size 10 \
  --beam-min-length 20 \
  --beam-block-ngram 3 \
  --beam-block-full-context True \
  --beam-context-block-ngram 3 \
  --input-path ./assets/bst_test_dialogues_200.json \
  --result-save-path ./results/blender9B_results.jsonl \
