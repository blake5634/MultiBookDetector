
import cv2
import numpy as np
import glob as gb
from book_hough import *
import newfcns as nf
import matplotlib.pyplot as plt
'''
Test some new functions for 
Find books in a bookshelf image


'''

scale = 3
img_width = int(3000/scale)
img_height =  int(4000/scale)
# cropped to one bookshelf:
img_width = int(1671)
img_height =  int(1206)


img_paths = gb.glob("tiny/*.jpg")
if (len(img_paths) < 1):
    print('No files found')
    quit()
for pic_filename in img_paths:
    print('looking at '+pic_filename)
    #img_gray, img = pre_process(pic_filename)
    #cv2.IMREAD_GRAYSCALE
    
    #
    #  read in and blur the image
    #
    img1 = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
 
    b=17
    assert b%2 != 0, ' blur radius (b) must be ODD'
    [img2, lab_image] = nf.KM(cv2.GaussianBlur(img1, (b,b), 0),10)
    
    
    ##############################################
    #
    #  Image Scanning parameters
    #
    # define multiple trancepts for robustness
    #
    trans_line_h = int(0.80*img_height)
    trans_line_dh = int(0.14*img_height)
    
    bar_thickness = 60 # how many pixels to study at each x value
    Ntrans = 4
    
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
    
        #print('shape vertvalues:',np.shape(vertvalues))
        fig, ax = plt.subplots()
        sline = nf.smooth(line,41)
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
        ########
        #plot sline on the image itself
        pm1 = tlh   # just draw lines up/down
        i=0
        A = 30
        for p in sline:
            r1 = tlh + int(A*pm1)
            c1 = i-1
            r2 = tlh + int(A*p)
            c2 = i
            #print('x1: {} y1: {} x2: {} y2: {}'.format(r1,c1,r2,c2))
            cv2.line(img2, (c1,r1), (c2,r2), (255,255,255), 2)
            i+=1
            pm1 = p
            
    # combined analysis of all vert bars:
    cres = [] # combined result at each point on line(s)
    print('iterating: ',np.shape(vvs[1][0]))
    #quit()
    for col in range(img_width): # go through the 
        data_x = []
        
        for line in range(Ntrans):
        #for v in vvs: # for each transcept values
            #print('\n\nsizes: v {} vvs {}\n\n'.format(np.shape(v), np.shape(vvs)))
            for t in vvs[line][col]:
                data_x.append(t)
        (val,cnts) = np.unique(data_x, return_counts=True)
        cres.append(val[np.argmax(cnts)]) # return most common label in the vert bar
        
    #
    #   plot a graph of the final combined, smoothed trancept
    fig, ax = plt.subplots() 
    sline = nf.smooth(cres,21)
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
    f.close
                       
    
    
    if True:
        #cv2.imshow("Raw",img1)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        cv2.imshow("Segmented", img2)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
    plt.show()
