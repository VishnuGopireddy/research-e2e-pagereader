######## SET DATASET PATH ###################

############## 3 in 1 ######################
python3 train.py --csv_train datasets/$1/train.csv\
        --csv_val ../datasets/$1/test.csv\
        --csv_classes ../datasets/$1/classes.csv\
		--score_threshold 0.35 \
		--early_stop_crit cer \
        --epochs_only_det 2\
        --train_htr True \
		--train_det True \
        --ner_branch False\
        --model_out $1"_3in1_"\
        --max_epochs_no_improvement 200\
        --seg_level word\
        --max_boxes 600
