#########################################################################
# File Name: run.sh
# Method: 
# Author: Jerry Shi
# Mail: jerryshi0110@gmail.com
# Created Time: 2019年11月04日 星期一 14时53分27秒
#########################################################################
#!/bin/bash

CUDA_VISIABLE_DEVICES="2,3"

DATA_DIR="/ftp_samba/cephfs/algo_data/yw.shi/dataset/short_docs/"
RESULT_DIR=$DATA_DIR/result/

echo $RESULT_DIR
# run
python main.py --train_data $DATA_DIR/doc_train.txt \
    --valid_data $DATA_DIR/doc_valid.txt    \
    --predict_data $DATA_DIR/doc_test.txt    \
    --stopwords $DATA_DIR/stopwords.txt \
    --model_dir $RESULT_DIR/model/   \
    --tensorboard_dir   $RESULT_DIR/tensorboard/    \
    --mode  "train" \
    --epoch 2
