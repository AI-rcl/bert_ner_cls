export MODEL_DIR=/home/rcl/PycharmProjects/myproject/test_project/bert_cls/cls_base/model_data
export BERT_BASE_DIR=/home/rcl/PycharmProjects/myproject/test_project/bert_cls/cls_base/chinese_L-12_H-768_A-12
python ckpt2pd.py \
    -bert_model_dir $BERT_BASE_DIR \
    -model_dir $MODEL_DIR \
    -max_seq_len 56 \
    -num_labels 4

