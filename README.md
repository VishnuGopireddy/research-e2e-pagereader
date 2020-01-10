# pytorch End-to-End Page Reader

Code for the papers [End-To-End Handwritten Text Detection and Transcription in Full Pages](http://www.cvc.uab.es/people/mcarbonell/papers/wml.pdf) and  [TreyNet: A Neural Model for Text Localization, Transcription and Named EntityRecognition in Full Pages](https://arxiv.org/pdf/1912.10016.pdf).

## Installation

1) Clone this repo

2) Install the required packages:

```
apt-get install tk-dev python-tk
```

3) Install the python packages:

```
pip3 install cffi

pip3 install cython

pip3 install opencv-python

pip3 install requests

pip3 install editdistance
```
Warp ctc pytorch: https://github.com/SeanNaren/warp-ctc

Python pagexml: https://github.com/omni-us/pagexml/tree/master/py-pagexml

## Detection and transcription on IAM:

### Prepare data
Download and extract IAM images and ground truth. You first need to register to the IAM database and set the environment variables for its credentials.

```
cd datasets/iam

IAM_U=[your IAMDB user]
IAM_PW=[your IAMDB password]

wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/forms/formsA-D.tgz --user $IAM_U --password $IAM_PW
wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/forms/formsE-H.tgz --user $IAM_U --password $IAM_PW
wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/forms/formsI-Z.tgz --user $IAM_U --password $IAM_PW

mkdir forms

tar -C forms -zxvf formsA-D.tgz
tar -C forms -zxvf formsE-H.tgz
tar -C forms -zxvf formsI-Z.tgz

wget http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/words.txt --user $IAM_U --password $IAM_PW
```
Get Aachen train, valid and test partitions
```
wget https://raw.githubusercontent.com/jpuigcerver/PyLaia/refactor_kws_egs_master/egs/iam-htr/data/part/forms/aachen/tr.lst

wget https://raw.githubusercontent.com/jpuigcerver/PyLaia/refactor_kws_egs_master/egs/iam-htr/data/part/forms/aachen/te.lst

wget https://raw.githubusercontent.com/jpuigcerver/PyLaia/refactor_kws_egs_master/egs/iam-htr/data/part/forms/aachen/va.lst

python generate_gt_IAM_csv.py words.txt

cd ../..

```


### Training, testing and predictions visualization


```
./experiments/set_path.sh

./train_e2e.sh

./test_e2e.sh

./predict_e2e.sh

```

### Alternatively trained model can be downloaded

```
mkdir trained_models
wget  -O trained_models/iam_join_det_htr_csv_retinanet.pt https://github.com/manucarbonell/models/blob/master/research-e2e-pagereader/iam_join_det_htr_lstm_csv_retinanet.pt?raw=true

./predict_e2e.sh

```
## Detection, transcription and named entity recognition on IEHHR:


### Prepare data
Download and extract IEHHR images and ground truth from https://rrc.cvc.uab.es/?ch=10&com=downloads.

Convert pagexml ground truth to csv to be read by the dataloader.
```
python3 pagexml2csv.py --pxml_dir datasets/esposalles/train --fout datasets/esposalles/train.csv --classes_out classes.csv --get_property True --seg_lev TextLine
python3 pagexml2csv.py --pxml_dir datasets/esposalles/valid --fout datasets/esposalles/valid.csv --classes_out classes.csv --get_property True --seg_lev TextLine
python3 pagexml2csv.py --pxml_dir datasets/esposalles/test --fout datasets/esposalles/test.csv --classes_out classes.csv --get_property True --seg_lev TextLine

```

Run experiment
```
./experiments/run.sh esposalles

```
# models
