import sys
import numpy as np
import pdb
from sklearn.cluster import KMeans
import cv2

fin = sys.argv[1]
f = open(fin,'r')
flines = f.readlines()
pdb.set_trace()
all_wh=np.zeros((len(flines),2))
prev_im=""

for line_idx in range(len(flines)):
    line = flines[line_idx]
    vals = line.split(',')
    x0 =int(vals[1])
    x1 =int(vals[3])
    y0 =int(vals[2])
    y1 =int(vals[4])
    tag=vals[5]
    text=vals[6].strip()
    im_file=vals[0]
    gt_im_path = im_file.split('.')[0]+'_gt.jpg'
    if im_file != prev_im:
        if line_idx>0:
            cv2.imwrite(gt_im_path,im)
            pdb.set_trace()
        im = cv2.imread(im_file)

    cv2.rectangle(im,(x0,y0),(x1,y1),(0,0,0))
    cv2.putText(im,text,(x0,y0),cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
    cv2.putText(im,tag,(x1,y1),cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

    prev_im=im_file

