#!/usr/bin/env bash
# @Author: bo.shi
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bo.shi
# @Last Modified time: 2020-01-01 11:46:07

TASK_NAME="tnews"
#MODEL_NAME="bert-base-chinese"
#MODEL_NAME="./prev_trained_model/electra_tnews_adapted_traintxt"
MODEL_NAME="./prev_trained_model/electra_small"
#/pytorch_model.bin
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export BERT_WWM_DIR=$BERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/CLUEdatasets
#
## download and unzip dataset
#if [ ! -d $GLUE_DATA_DIR ]; then
#  mkdir -p $GLUE_DATA_DIR
#  echo "makedir $GLUE_DATA_DIR"
#fi
#cd $GLUE_DATA_DIR
#if [ ! -d $TASK_NAME ]; then
#  mkdir $TASK_NAME
#  echo "makedir $GLUE_DATA_DIR/$TASK_NAME"
#fi
#cd $TASK_NAME
#if [ ! -f "train.json" ] || [ ! -f "dev.json" ] || [ ! -f "test.json" ]; then
#  rm *
#  wget https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip
#  unzip tnews_public.zip
#  rm tnews_public.zip
#else
#  echo "data exists"
#fi
#echo "Finish download dataset."

# make output dir
if [ ! -d $CURRENT_DIR/milnews_output ]; then
  mkdir -p $CURRENT_DIR/milnews_output
  echo "makedir $CURRENT_DIR/milnews_output"
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
if [ $# == 0 ]; then
    python run_classifier.py \
      --model_type=electra \
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir=$GLUE_DATA_DIR/milnews/ \
      --max_seq_length=128 \
      --per_gpu_train_batch_size=16 \
      --per_gpu_eval_batch_size=16 \
      --learning_rate=2e-5 \
      --num_train_epochs=3 \
      --logging_steps=3335 \
      --save_steps=3335 \
      --output_dir=$CURRENT_DIR/milnews_output/ \
      --overwrite_output_dir \
      --seed=42
fi
