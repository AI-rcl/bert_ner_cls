#!/usr/bin/env bash
export BERT_BATH_DIR='/home/rcl/PycharmProjects/myproject/test_project/bert_cls/cls_base/chinese_L-12_H-768_A-12'
export DATA_DIR='/home/rcl/PycharmProjects/myproject/test_project/bert_cls/cls_base/ner001'
export OUTPUT_DIR='/home/rcl/PycharmProjects/myproject/test_project/bert_cls/cls_base/model_data'

python bert_cls/cls_base/run_classifier.py \
  --data_dir=$DATA_DIR \
  --task_name=sim \
  --vocab_file=$BERT_BATH_DIR/vocab.txt \
  --bert_config_file=$BERT_BATH_DIR/bert_config.json \
  --output_dir=$OUTPUT_DIR \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=$BERT_BATH_DIR/bert_model.ckpt \
  --max_seq_length=56 \
  --train_batch_size=24 \
  --learning_rate=5e-5 \
  --num_train_epochs=10.0
