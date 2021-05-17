#!/bin/zsh
#!/bin/bash
set -eux

EXPERIMENT_DIR=$1
EXPERIMENT_BASE_NAME=${2:-experiment}
BATCH_SIZE=${3:-48}
OPTIMIZER=${4:-adamax}
LR=${5:-5e-05}
INIT_MODEL_PATH=${6:-zoo:pretrained_transformers/bi_model_huge_reddit/model}

EXPERIMENT_NAME="${EXPERIMENT_BASE_NAME}_batch${BATCH_SIZE}_opt-${OPTIMIZER}_lr-${LR}"
MODEL_PATH="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}"


python3 ParlAI/parlai/scripts/train_model.py \
  --init-model $INIT_MODEL_PATH \
  -t bst_distill \
  \
  `# Model params` \
  --model transformer/biencoder_distill \
  --output-scaling 0.06 \
  --variant xlm \
  --reduction-type mean \
  --share-encoders False \
  --learn-positional-embeddings True \
  --n-layers 12 \
  --n-heads 12 \
  --ffn-size 3072 \
  --attention-dropout 0.1 \
  --relu-dropout 0.0 \
  --dropout 0.1 \
  --n-positions 1024 \
  --embedding-size 768 \
  --activation gelu \
  --embeddings-scale False \
  --n-segments 2 \
  --learn-embeddings True \
  --share-word-embeddings False \
  --distill-loss-coef 0.0 \
  --use-cand-loss-mask \
  \
  `# Data params` \
  --data-parallel True \
  --history-size 20 \
  --label-truncate 128 \
  --text-truncate 240 \
  --candidates batch \
  --train-data-postfix g2r-ll \
  --num-random-negatives 512 \
  --response-path ./datasets/emnlp_2021_g2r_dataset/bst_data_level_g2r_responses.txt \
  \
  `# Tokenizer params` \
  --dict-file zoo:pretrained_transformers/bi_model_huge_reddit/model.dict \
  --dict-tokenizer bpe \
  --dict-lower True \
  --dict-endtoken __start__ \
  \
  `# Training params` \
  `## Batch related` \
  \
  --batchsize ${BATCH_SIZE} \
  --eval-batchsize 12 \
  --eval-candidates inline \
  \
  `## Learning Rate` \
  --optimizer ${OPTIMIZER} \
  -lr ${LR} \
  --lr-scheduler reduceonplateau \
  --lr-scheduler-patience 1 \
  --gradient-clip 0.1 \
  --warmup-updates 100 \
  --optimizer-hard-reset True \
  \
  `## Training time` \
  --num-epochs 50.0 \
  --max-train-time 200000 \
  \
  `# Validation` \
  -veps 0.333 \
  -vme 80000 \
  --validation-metric "hits@1" \
  --validation-metric-mode max \
  --save-after-valid True \
  --log-every-n-secs 20 \
  \
  `# FP16` \
  --fp16 True \
  \
  `# TB Logging` \
  --tensorboard-log True \
  \
  `# Model` \
  --model-file ${MODEL_PATH}
