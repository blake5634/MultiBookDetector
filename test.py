
import cv2
import numpy as np
import glob as gb
from book_hough import *
import newfcns as nf
import book_parms as bpar
import matplotlib.pyplot as plt
'''
Test some new functions for 
Find books in a bookshelf image


''' 
# cropped to one bookshelf:
img_width = int(1671)
img_height =  int(1206)

PLOTS = False

img_paths = gb.glob("tiny/*.jpg")

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
    img = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
    ish = img.shape
    #
    #  scale the image 
    #
    #     scale factor imported from newfcns.py
    #
        
    img_width = int(ish[1]/bpar.scale)
    img_height =  int(ish[0]/bpar.scale)
    img1 = cv2.resize(img, (img_width, img_height))
    
    ############
    #
    #   blur and K-means cluster 
    #
    
    b=17
    assert b%2 != 0, ' blur radius (b) must be ODD'
    [img2, lab_image] = nf.KM(cv2.GaussianBlur(img1, (b,b), 0),10)
        
    ##########################3
    #
    #   Look across top of image for label of "black"
    #
    bls = []
    for col in range(img_width):
        for r in range(10):
            bls.append(lab_image[r+5,col])
    labs, cnt = np.unique(bls, return_counts=True)
    max = np.max(cnt)
    for l,c in zip(labs,cnt):
        if c == max:
            lmax = l
            break
    print('Most common label near top: {} ({}%)'.format(lmax, 100*max/np.sum(cnt)))
    
    f = open('metadata.txt','w')
    print('dark label, {}'.format(lmax), file=f)
    f.close()
    
    ##############################################
    #
    #  Image Scanning parameters
    #
    # define multiple trancepts for robustness
    #
    
    trans_line_h = int(0.80*img_height)
    trans_line_dh = int(0.14*img_height)
    
    bar_thickness = int(60/bpar.scale) # how many pixels to study at each x value
    Ntrans = 3
    
    assert trans_line_dh * Ntrans < trans_line_h, 'Illegal horizontal/vertical scan params'
    
    sls = []  # store smoothed label lines
    vvs = []  # store individual pixel value arrays for each line
    
    #
    #   Iterate through transcept lines
    #
    for i in range(Ntrans): # perform Ntrans trancepts
        tlh = trans_line_h - int(i*trans_line_dh)   # go up the image 
        assert tlh > 0 + bar_thickness/2, 'a transcept line is too close to top'
        
        #draw the trancept line
        #cv2.line(new_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.line(img2, (0,tlh), (img_width-1, tlh), (255,0,0), 2)

        #line = nf.Trancept_labeled(lab_image, tlh)
        line, vertvalues= nf.Trancept_bar_labeled(lab_image, tlh, bar_thickness)
    
        sline = nf.smooth(line,41)
        if PLOTS:
            #print('shape vertvalues:',np.shape(vertvalues))
            fig, ax = plt.subplots()
            #print(':::: sline: ',type(sline), np.shape(sline))
            ax = plt.plot(range(len(sline)),sline)
        sls.append(sline)       # store the smoothed label line
        vvs.append(vertvalues)  # store all labels in vertical bars
        
        ##################################
        #
        #  plot analysis window on the image
        #
        ymin = tlh - int(bar_thickness/2)
        ymax = tlh + int(bar_thickness/2)
        x1 = 0
        x2 = img_width-1
        cv2.line(img2, (x1,ymin),(x2,ymin), (0,255,0), 2)
        cv2.line(img2, (x1,ymax),(x2,ymax), (0,255,0), 2)
        
    #
    #  Analyze the trancept data
    #
    # for all computed lines
    for sline in sls:
        ###########################################################
        #
        #     plot smoothed lines on the image itself
        #
        pm1 = tlh   # just draw lines up/down
        i=0
        A = 30/bpar.scale
        for p in sline:
            r1 = tlh + int(A*pm1)  #'y1'
            c1 = i-1
            r2 = tlh + int(A*p)    #'y2'
            c2 = i
            #print('x1: {} y1: {} x2: {} y2: {}'.format(r1,c1,r2,c2))
            cv2.line(img2, (c1,r1), (c2,r2), (255,255,255), 2)
            i+=1
            pm1 = p
    ###############################################################
    #
    # combined analysis of all vert bars:
    #
    cres = [] # combined result at each point on line(s)
    print('iterating: ',np.shape(vvs[1][0]))
    
    for col in range(img_width): # go through the 
        data_x = []
        
        for line in range(Ntrans):
        #for v in vvs: # for each transcept values
            #print('\n\nsizes: v {} vvs {}\n\n'.format(np.shape(v), np.shape(vvs)))
            for t in vvs[line][col]:
                data_x.append(t)
        (val,cnts) = np.unique(data_x, return_counts=True)
        cres.append(val[np.argmax(cnts)]) # return most common label in the vert bar
        
    print('Final combined line length: {}'.format(len(cres)))
    #
    #   plot a graph of the final combined, smoothed trancept
    fig, ax = plt.subplots()
    
    #  smooth_size defined in newfcns
    
    swinsize = int(bpar.smooth_size/bpar.scale)
    if swinsize > 0:
        if swinsize%2==0:  # must be ODD
            swinsize += 1
        sline = nf.smooth(cres,swinsize)
        print('Final *smoothed* line length: {}'.format(len(sline)))
    else:
        print('no smoothing of label data')
        sline = cres

    #print(':::: sline: ',type(sline), np.shape(sline))
    ax = plt.plot(range(len(sline)),sline)
    #ax = plt.plot(range(len(nf.smooth()),sline))
    plt.title('Combined all N trancepts')
    
    #
    #  save the data for quicker testing
    #
    f = open('ctrans_labels.csv','w')
    for i in range(len(sline)):
        print('{}, {:8.3f}'.format(i,sline[i]),file=f)
        #print('{}, {:8.3f}'.format(i,sline[i]))
    f.close()
                       
    
    
    if True:
        #cv2.imshow("Raw",img1)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        cv2.imshow("Segmented", img2)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    plt.show()
