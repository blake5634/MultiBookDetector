
import cv2
import numpy as np
import time as time
import glob as gb
from book_hough import *
import newfcns as nf
import matplotlib.pyplot as plt
'''
Test some new functions for 
Find books in a bookshelf image


 **********  look for major lines at different angles
 
''' 
 

img_paths = gb.glob('tiny/target.jpg')
if (len(img_paths) < 1):
    print('No files found')
    quit()
for pic_filename in img_paths:
    print('looking at '+pic_filename)
    #img_gray, img = pre_process(pic_filename)
    #cv2.IMREAD_GRAYSCALE
    
    #
    #  read in the image
    #
    #img = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
    img_orig = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
    ish = img_orig.shape
    #
    #  scale the image 
    #
    #     scale factor imported from newfcns.py
    #
        
    img_width = int(ish[1]/nf.scale)
    img_height =  int(ish[0]/nf.scale)
    img1 = cv2.resize(img_orig, (img_width, img_height))
            
        
    ############
    #
    #   blur  
    #
    
    b = int(nf.blur_rad/nf.scale)
    if b%2 == 0:
        b+=1
    img2 = cv2.GaussianBlur(img1, (b,b), 0)
        

    ############
    #
    #  Use KMeans to posterize to N color labels
    #
    N = nf.KM_Clusters
    img0, label_img, ctrs = nf.KM(img2,N)   
    #cv2.imshow("labeled KM image", img0)
    imct = nf.Gen_cluster_colors(ctrs)
    cv2.imshow("cluster colors (10 max)",imct)
    cv2.waitKey(1000)
    nfound = 0

    #
    #   Find background label
    #
    backgnd = nf.Check_background(label_img)
    
    #
    #  look for lines at a bunch of angles
    #
    sl_wi = 10
    for th in [95, 120, 150, 180]:
        for xint in range(50, img_width, 5):
            for col in range(200,img_width):
                lscore = nf.Get_line_score(img0, col, sl_wi, xint, th)
                print('Col: {} score: {}'.format(col, lscore))
        
        
        
        
    print('{}, {} booktangles detected'.format(pic_filename, nfound) )

    cv2.imshow(title, img_orig)
    cv2.waitKey(-1)
