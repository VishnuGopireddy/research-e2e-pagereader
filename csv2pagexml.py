import pagexml
import sys
import numpy as np
import os
import pdb
import cv2

fin = sys.argv[1]
f = open(fin,'r')
flines = f.readlines()
pdb.set_trace()
all_wh=np.zeros((len(flines),2))
prev_im=""
if not os.path.exists('pagexmls'):
    os.mkdir('pagexmls')

pxml = pagexml.PageXML()

for line_idx in range(len(flines)):
    line = flines[line_idx]
    vals = line.split(',')
    x0 =int(vals[1])
    x1 =int(vals[3])
    y0 =int(vals[2])
    y1 =int(vals[4])
    width=x1-x0
    height=y1-y0
    tag=vals[5]
    text=vals[6].strip()
    im_file=vals[0]
    image_id = im_file.split('.')[-2].split('/')[-1]
    gt_im_path = im_file.split('.')[0]+'_gt.jpg'
    gtxml_name = im_file.split('.')[0]+'.xml'

    if im_file != prev_im:
        box_id=0
        if line_idx>0:
            pdb.set_trace()
            pxml.write('pagexmls/'+image_id+".xml")
        im=cv2.imread(im_file)
        pxml.newXml('pages',im_file,im.shape[1],im.shape[0])
        page = pxml.selectNth("//_:Page",0)
        reg = pxml.addTextRegion(page)

        pxml.setCoordsBBox(reg,0, 0, width, height)
        textline = pxml.addTextLine(reg)
        pxml.setCoordsBBox(textline,0, 0, width, height)
        words = []

    # Add a text region to the Page
    word = pxml.addWord(textline,"ID"+str(box_id))
    
    # Set text region bounding box with a confidence
    pxml.setCoordsBBox(word,x0, y0, x1-x0, y1-y0)
    pxml.setTextEquiv(word, text )

    # Add property to text region
    pxml.setProperty(word,"category" , tag )

    # Add a second page with a text region and specific id
    #page = pxml.addPage("example_image_2.jpg", 300, 300)
    #reg = pxml.addTextRegion( page, "regA" )
    #pxml.setCoordsBBox( reg, 15, 12, 76, 128 )
    words.append(word)
    box_id+=1
    prev_im=im_file

