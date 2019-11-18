######## SET DATASET PATH ###################
DATASET=$1


############## 3 in 1 ######################
python3 get_pagexmls.py --csv_val ../../DATASETS/$DATASET/test.csv\
        --csv_classes ../../DATASETS/$DATASET/classes.csv\
		--score_threshold 0.35 \
        --model "trained_models/"$DATASET"_3in1_csv_retinanet.pt"

############ 3 in 1 seq #################

python3 get_pagexmls.py --csv_val ../../DATASETS/$DATASET/test.csv\
        --csv_classes ../../DATASETS/$DATASET/classes.csv\
		--score_threshold 0.35 \
        --model "trained_models/"$DATASET"_3in1_seq_csv_retinanet.pt"\

############# det + ner seq #################

python3 get_pagexmls.py  --csv_val ../../DATASETS/$DATASET/test.csv\
        --csv_classes ../../DATASETS/$DATASET/classes.csv\
		--score_threshold 0.35 \
        --model "trained_models/"$DATASET"_det_ner_seq_csv_retinanet.pt"\

###### DET + HTR #############################


python3 get_pagexmls.py  --csv_val ../../DATASETS/$DATASET/test.csv\
        --csv_classes ../../DATASETS/$DATASET/classes.csv\
		--score_threshold 0.35 \
        --model "trained_models/"$DATASET"_det_htr_csv_retinanet.pt"\

