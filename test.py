import editdistance
import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import csv_eval_binary_map 
import csv_eval

from get_transcript import get_transcript

from warpctc_pytorch import CTCLoss


print(('CUDA available: {}'.format(torch.cuda.is_available())))


def main(args=None):

    parser     = argparse.ArgumentParser(description='Simple testing script for RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.',default = "csv")
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)',default="binary_class.csv")
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--csv_box_annot', help='Path to file containing predicted box annotations ')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=500)
    parser.add_argument('--model', help='Path of .pt file with trained model',default = 'esposallescsv_retinanet_0.pt')
    parser.add_argument('--model_out', help='Path of .pt file with trained model to save',default = 'trained')

    parser.add_argument('--score_threshold', help='Score above which boxes are kept',default=0.15)
    parser.add_argument('--nms_threshold', help='Score above which boxes are kept',default=0.2)
    parser.add_argument('--max_epochs_no_improvement', help='Max epochs without improvement',default=100)
    parser.add_argument('--max_boxes', help='Max boxes to be fed to recognition',default=50)
    parser.add_argument('--seg_level', help='Line or word, to choose anchor aspect ratio',default='line')
    parser.add_argument('--htr_gt_box',help='Train recognition branch with box gt (for debugging)',default=False)
    parser.add_argument('--binary_classifier',help='Wether to use classification branch as binary or not, multiclass instead.',default='False')
    parser = parser.parse_args(args)
    
    # Create the data loaders

    if parser.dataset == 'csv':


        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')


        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))


        if parser.csv_box_annot is not None:
            box_annot_data = CSVDataset(train_file=parser.csv_box_annot, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

        else:    
            box_annot_data = None
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    if box_annot_data is not None:
        sampler_val = AspectRatioBasedSampler(box_annot_data, batch_size=1, drop_last=False)
        dataloader_box_annot = DataLoader(box_annot_data, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    else:
        dataloader_box_annot = dataloader_val

    if not os.path.exists('trained_models'):
        os.mkdir('trained_models')

    # Create the model

    alphabet=dataset_val.alphabet
    if os.path.exists(parser.model):
        retinanet = torch.load(parser.model)
    else:
        print("Choose an existing saved model path.")
        sys.exit()
    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()
    
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    
    #retinanet = torch.load('../Documents/TRAINED_MODELS/pytorch-retinanet/esposallescsv_retinanet_99.pt')
    #print "LOADED pretrained MODEL\n\n"
    

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    ctc = CTCLoss()
    retinanet.module.freeze_bn()
    best_cer = 1000
    epochs_no_improvement=0
    
    cers=[]    
    retinanet.eval()
    retinanet.module.epochs_only_det = 0
    #retinanet.module.htr_gt_box = False
    
    retinanet.training=False    
    if parser.score_threshold is not None:
        retinanet.module.score_threshold = float(parser.score_threshold) 
    
    '''if parser.dataset == 'csv' and parser.csv_val is not None:

        print('Evaluating dataset')
    '''
    mAP,cer = csv_eval.evaluate(dataset_val, retinanet,score_threshold=retinanet.module.score_threshold)
    print ("VALID NER mAP:",mAP)

    det_map,cer= csv_eval_binary_map.evaluate(dataset_val, retinanet,score_threshold=retinanet.module.score_threshold)
    print ("VALID text det mAP:",det_map)

if __name__ == '__main__':
 main()
