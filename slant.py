
import cv2
import numpy as np
import time as time
import glob as gb
from book_hough import *
import newfcns as nf
import book_parms as bpar
import matplotlib.pyplot as plt
'''
Test some new functions for 
Find books in a bookshelf image


 **********  look for major lines at different angles
 
''' 
 

img_paths = gb.glob('tiny/target.jpg')
d2r = 2*np.pi/360.0  #deg to rad

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
    
    #############   
    #
    #  Standardize the image sizes/scales
    #
    
    orig_ish, scaled_ish = nf.Get_sizes(img_orig, bpar.scale)
    
    #
    #
    #  scale the image 
    #
    #     scale factor imported from newfcns.py
    #
        
    img_width = scaled_ish[1]
    img_height = scaled_ish[0] 
    img1 = cv2.resize(img_orig, (img_width, img_height))
    
            
        
    ############
    #
    #   blur  
    #
    
    b = int(bpar.blur_rad/bpar.scale)
    if b%2 == 0:
        b+=1
    img2 = cv2.GaussianBlur(img1, (b,b), 0)
        

    ############
    #
    #  Use KMeans to posterize to N color labels
    #
    N = bpar.KM_Clusters
    img0, label_img, ctrs, color_dist = nf.KM(img2,N)   
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
    #  look for lines at a bunch of x-values
    #

    # book shape parameters target
    sl_wi = bpar.slice_width #mm # slice window (dist above/below (90deg to line) line) 
    length = bpar.book_edge_line_length_mm  # mm

    linescanx_mm = range(160, -160, -2) #mm
    ang_scan_deg = range(110,170,2)  # deg 
    
    lines_found = []
    
    
    for xmm in linescanx_mm:
        for th in ang_scan_deg:
    #if True:
            #(th, xmm) = (140, -40)
            # XY coordinates relative to (img_width/2, img_height/2) +=up
            
            # some test values 
            #x1 = 88.0 #mm  black yellow book bdry
            #x1 = 102.0 #mm  middle of tan book(!)
            #x1 = 65.0 #mm  middle of blue book(!)
            #x1 = 60.0 #mm   btwn blue and black book
            
            x1 = xmm
            
            #th = 145   # deg relative to 03:00 (clock)
            #th = 142  # deg relative to 03:00 (clock)
            m0 = np.tan(th*d2r) # slope (mm/mm)
            #print('---Shape label image: {}'.format(np.shape(label_img)))
            #print('Image sample: {}'.format(label_img[10,10]))

            #
            #     Get the line score
            #
            #lscore = 0.99999999
            lscore = nf.Get_line_score(label_img, sl_wi, x1, th, length, bpar.line_bias, color_dist)  # x=0, th=125deg
            print('X: {} th: {} score: {}'.format(x1, th, lscore))

            if lscore > bpar.Line_Score_Thresh:
                print('line at {:4.2f} is important, score ({})'.format(xmm,lscore))
                lines_found.append((xmm,th,lscore))
        
    # 
    # tsti will be the desired display image (tstscale is IT's scale) 
    tsti = img_orig.copy()  # original 
    tstscale = 1
    #tstscale = 3  #????
    #tsti = img0      # scaled and VQ'd by KM()
    #tstscale = 3
    
    (xmin, xmax, ymin, ymax) = nf.Get_mmBounds(tsti,iscale=tstscale)


    print('xmin,xmax,ymin,ymax:  {} {} {} {}'.format(xmin,xmax,ymin,ymax))
    
    print('found {} lines initially'.format(len(lines_found)))
    if True:
        #
        # cluster the lines found.   Get strongest line in each bunch
        #
        maxgap = bpar.max_gap_mm #mm   biggest gap inside a bunch
        strong_lines = []
        xp = -500 #mm
        lsmax = -99  # max score in a bunch
        xsmax = -200
        thmax = -200
        first = True
        for x,th,score in lines_found:
            if abs(x-xp) > maxgap and not first: # mm
                print('New Strong Line: x:{} xsmax:{}'.format(x,xsmax))
                strong_lines.append((xsmax,thmax, lsmax))
                lsmax = -99
                xp = x
                next
            if score > lsmax:
                lsmax = score
                xsmax = x
                thmax = th
            first = False
            xp = x
        print('found {} strong lines'.format(len(strong_lines)))
        #print ('strongest lines:')
        #print (strong_lines)
    else:
        strong_lines = lines_found
        
    for x1,th,score in strong_lines:
    #
    #   Draw the testing line and bounds 
    #
        d2r = 2*np.pi/360.0  #deg to rad
        m0 = np.tan(th*d2r) # slope (mm/mm)
        b0 = -m0*x1  # mm 
        rV = sl_wi/np.cos((180-th)*d2r)  # mm
        #rVp, dummy  = nf.XY2RC(tsti,rV,0)  # pixels
        rVp = int(rV * bpar.mm2pix)     # pixels

        #print('m0: {} b0: {}mm rp:{}(pix)'.format(m0,b0,rp))
        #print('th: {} deg, iw {}  ih: {}'.format(th,iw,ih))
        dx = abs((length/2)*np.cos(th*d2r)) # mm
        xmin2 = x1 - dx  #mm    X range for test line
        xmax2 = x1 + dx  #mm
        # cols,  rows = XY2iXiY()
        xmi2p, dummy = nf.XY2iXiY(tsti, xmin2,0)  # pix  X range for test line
        xmx2p, dummy = nf.XY2iXiY(tsti, xmax2,0)
        rng = range(xmi2p, xmx2p-1, 1)  # pix cols
        # the line
        colcode='yellow'
        if score > 1.1*bpar.Line_Score_Thresh:
            colcode = 'green'
        if score > 1.2*bpar.Line_Score_Thresh:
            colcode = 'blue'
        if score > 1.6*bpar.Line_Score_Thresh:
            colcode = 'red'
        if score > 2.0*bpar.Line_Score_Thresh:
            colcode = 'white'
        colcode = 'yellow'
        nf.DLine_mm(tsti, (xmin2, bpar.line_bias + m0*xmin2+b0), (xmax2, bpar.line_bias + m0*xmax2+b0), colcode,iscale=tstscale)
        # above window line
        #nf.DLine_mm(tsti, (xmin2,  rV + m0*xmin2+b0), (xmax2,  rV + m0*xmax2+b0), 'blue',iscale=tstscale)
        #nf.DLine_mm(tsti, (xmin2, -rV + m0*xmin2+b0), (xmax2, -rV + m0*xmax2+b0), 'green',iscale=tstscale)
        
  
  
  
    ###################################################################3
    #
    #  Draw some debugging graphics
    #
    # Draw H and V axes (X,Y axes in mm)
    nf.DLine_mm(tsti, (xmin,0), (xmax,0),'white',iscale=tstscale)
    nf.DLine_mm(tsti, (0, ymin), (0, ymax), 'white',iscale=tstscale)

    ## Draw some tick marks
    tick_locs_mm = [] # pix
    tickwidth = 20 # mm
    for xt in range(15): # unitless
        xpt = tickwidth*(xt+1)  # mm
        tick_locs_mm.append(xpt)
        tick_locs_mm.append(-xpt)
    ya = 0.0 #mm
    yb = -5.0 #mm
    for x in tick_locs_mm:
        nf.DLine_mm(tsti, (x, ya), (x,yb), 'green',iscale=tstscale)



    if False:
        # Draw the line for debugging
        d2r = 2*np.pi/360.0  #deg to rad
        m0 = np.tan(th*d2r)
        b0 = -m0*x1    # mm
        x = x1         # mm
        dx = abs((length/2)*np.cos(th*d2r))  #mm
        xa, ya = nf.XY2iXiY(tsti, x-dx, m0*(x-dx)+b0)  # parms in mm
        xb, yb = nf.XY2iXiY(tsti, x+dx, m0*(x+dx)+b0)
        print('Image shape: {}'.format(np.shape(tsti)))
        print('pt a: {},{},  pt b: {},{}'.format(xa,ya,xb,yb))
        ir = np.shape(tsti)[0]  # image rows
        ic = np.shape(tsti)[1]  # image cols
        cv2.line(tsti, (xa,ya), (xb,yb), (255,255,255), 3)

        # line window border lines
        r = int(nf.mm2pix*sl_wi/np.cos((180-th)*d2r))  # pix
        ya += r
        yb += r
        cv2.line(tsti, (xa,ya), (xb,yb), (0,0,255), 2)
        ya -= 2*r
        yb -= 2*r
        cv2.line(tsti, (xa,ya), (xb,yb), (0,0,255), 2)
 
    
    print('{}, {} booktangles detected'.format(pic_filename, nfound) )


    title='test image'
    cv2.imshow(title, tsti)
    cv2.waitKey(-1)
