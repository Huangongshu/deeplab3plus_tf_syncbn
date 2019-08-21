set -e

# Move one-level up to tensorflow/models/research directory.
cd ../..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"
CS_FOLDER="cityscapes"
EXP_FOLDER="exp/xception71_dsp_baseline"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${CS_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CS_FOLDER}/${EXP_FOLDER}/train"
CS_DATASET="${WORK_DIR}/${DATASET_DIR}/${CS_FOLDER}/tfrecord"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${CS_FOLDER}/${EXP_FOLDER}/eval"

python "${WORK_DIR}"/train.py \
  --logtostderr \
  --training_number_of_steps=90000 \
  --train_split="train" \
  --model_variant="xception_71" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=769 \
  --train_crop_size=769 \
  --train_batch_size=8 \
  --tf_initial_checkpoint="${INIT_FOLDER}/xception_71/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${CS_DATASET}" \
  --num_clones=4 \
  --base_learning_rate=0.007 \
  --dataset="cityscapes" \
  --dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json" \

python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_71" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --decoder_output_stride=4 \
  --eval_crop_size=1025 \
  --eval_crop_size=2049 \
  --dataset="cityscapes" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${CS_DATASET}" \
  --max_number_of_evaluations=1 \
  --dense_prediction_cell_json="deeplab/core/dense_prediction_cell_branch5_top1_cityscapes.json"
