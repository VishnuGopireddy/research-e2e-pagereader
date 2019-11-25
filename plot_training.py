import sys
import os
import numpy as np
import matplotlib.pyplot as plt

dataset = sys.argv[1]
fin=os.path.join('trained_models',dataset+'_3in1_seq_log.txt')
train_log= open(fin,'r')
lines = train_log.readlines()
cers=[]
ner_maps=[]
det_maps=[]
epochs = []
for line in lines:
    vals = line.split(' ')
    if len(vals)!=6: continue
    epochs.append(int(vals[0]))
    
    cers.append(1-float(vals[1]))
    ner_maps.append(float(vals[3]))
    det_maps.append(float(vals[5]))
    
cers = np.array(cers)[:45]
ner_maps=np.array(ner_maps)[:45]
det_maps=np.array(det_maps)[:45]
epochs = np.array(epochs)[:45]
# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(epochs,det_maps, 'r',label='det mAP')
plt.plot(epochs,cers, 'b',label='HTR 1-CER')
plt.plot(epochs,ner_maps, 'g',label='NER mAP')
#plt.ylabel('mAP / 1-CER')
#plt.xlabel('epochs')
plt.legend()
#plt.show()
plt.savefig(dataset+'_train_log.png')
