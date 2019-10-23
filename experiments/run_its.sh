######## SET DATASET PATH ###################
DATASET='gt-its'

: '
############## 3 in 1 ######################
python3 train.py --csv_train ../../DATASETS/$DATASET/train.csv\
        --csv_val ../../DATASETS/$DATASET/test.csv\
        --csv_classes ../../DATASETS/$DATASET/classes.csv\
		--score_threshold 0.35 \
		--early_stop_crit cer \
        --epochs_only_det 2\
        --train_htr True \
		--train_det True \
        --ner_branch False\
        --model_out $DATASET"_3in1_"\
        --max_epochs_no_improvement 50\
        --seg_level line\
        --max_boxes 600
'
############# det + ner seq #################

python3 train.py --csv_train ../../DATASETS/$DATASET/train.csv\
        --csv_val ../../DATASETS/$DATASET/test.csv\
        --csv_classes ../../DATASETS/$DATASET/classes.csv\
		--score_threshold 0.35 \
		--early_stop_crit map \
        --epochs_only_det 2\
        --train_htr False \
		--train_det True \
        --ner_branch True\
        --model_out $DATASET"_det_ner_seq_"\
        --max_epochs_no_improvement 50\
        --seg_level line\
        --max_boxes 600
############ 3 in 1 seq #################

python3 train.py --csv_train ../../DATASETS/$DATASET/train.csv\
        --csv_val ../../DATASETS/$DATASET/test.csv\
        --csv_classes ../../DATASETS/$DATASET/classes.csv\
		--score_threshold 0.35 \
		--early_stop_crit cer \
        --epochs_only_det 2\
        --train_htr True \
		--train_det True \
        --ner_branch True\
        --model_out $DATASET"_3in1_seq_"\
        --max_epochs_no_improvement 50\
        --seg_level line\
        --max_boxes 600


###### DET + HTR #############################


python3 train.py --csv_train ../../DATASETS/$DATASET/train.csv\
        --csv_val ../../DATASETS/$DATASET/test.csv\
        --csv_classes ../../DATASETS/$DATASET/classes.csv\
		--score_threshold 0.35 \
		--early_stop_crit cer \
        --epochs_only_det 2\
        --train_htr True \
		--train_det True \
        --ner_branch False\
        --model_out $DATASET"_det_htr_"\
        --max_epochs_no_improvement 50\
        --seg_level line\
        --max_boxes 600\
        --binary_classifier True

