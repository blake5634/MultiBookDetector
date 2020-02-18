
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
    #print('label_img shape: {}'.format(np.shape(label_img)))
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
    sl_wi = 50  # slice window (dist above/below (90deg to line) line)
    #for th in [95, 120, 150, 180]:
        #for xicpt in range(0, img_width/2, 5):  # should be -iw/2 --- iw/2
            #lscore = nf.Get_line_score(label_img, sl_wi, xicpt, th)
            #print('X: {} th: {} score: {}'.format(xicpt, th, lscore))

    # XY coordinates relative to (img_width/2, img_height/2) +=up
    x1 = 32 #  black yellow book bdry
    x1 = int(65*nf.mm2pix*nf.scale)
    th = 145   # deg relative to 03:00 (clock)
    length = 250  #
    print('---Shape label image: {}'.format(np.shape(label_img)))
    print('Image sample: {}'.format(label_img[10,10]))

    #
    #     Get the line score
    #
    lscore = nf.Get_line_score(label_img, sl_wi, x1, th, length)  # x=0, th=125deg
    print('X: {} th: {} score: {}'.format(x1, th, lscore))        

    # Draw the line for debugging
    d2r = 2*np.pi/360.0  #deg to rad
    m0 = np.tan(th*d2r)
    b0 = -m0*x1
    x = x1
    dx = abs(int((length/2)*np.cos(th*d2r)))
    xa, ya = nf.XY2iXiY(img_orig, x-dx, int(m0*(x-dx)+b0))
    xb, yb = nf.XY2iXiY(img_orig, x+dx, int(m0*(x+dx)+b0))
    print('Image shape: {}'.format(np.shape(img_orig)))
    print('pt a: {},{},  pt b: {},{}'.format(xa,ya,xb,yb))
    ir = np.shape(img_orig)[0]  # image rows
    ic = np.shape(img_orig)[1]  # image cols
    cv2.line(img_orig, (xa,ya), (xb,yb), (255,255,255), 3)

    # line window border lines
    r = int(sl_wi/np.cos((180-th)*d2r))
    ya += r
    yb += r
    cv2.line(img_orig, (xa,ya), (xb,yb), (0,0,255), 2)
    ya -= 2*r
    yb -= 2*r
    cv2.line(img_orig, (xa,ya), (xb,yb), (0,0,255), 2)

    # horizontal axis
    cv2.line(img_orig, (0,int(ir/2)), (ic,int(ir/2)), (255,255,255), 2)
    # vertical axis
    cv2.line(img_orig, (int(ic/2),0), (int(ic/2),ir), (255,255,255), 2)
    
    # 1 cm tick marks
    row = int(ir/2)
    col = int(6*nf.mm2pix*nf.scale)  # line up with x=0
    ya = int(ir/2)
    tick = int(5*nf.mm2pix*nf.scale)
    yb = int(ir/2 + tick)
    while col < ic:
        xa = col
        cv2.line(img_orig, (xa,ya), (xa,yb), (255,255,255), 2)
        col += int(10*nf.mm2pix*nf.scale)
    
    
    print('{}, {} booktangles detected'.format(pic_filename, nfound) )


    title='original image'
    cv2.imshow(title, img_orig)
    cv2.waitKey(-1)
