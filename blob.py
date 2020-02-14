
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


 **********   use SimpleBlobDetector()

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
    
    
          
    ##############
    #
    #    Init SimpleBlobDetector
    #
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 200
    params.maxThreshold = 255
    # set to look for bright blobs
    params.blobColor=255
    # Filter by Area.
    params.filterByArea = True
    params.minArea = int(700/nf.scale)
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.5
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
        
        
        
        
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
    
    for lab in range(N):
        if lab == background:
            print ('skipping background label: {}'.format(backgnd))
            continue
        imgx = (label_img==lab).astype(np.uint8)  # change True to 1.0
        imgx = 255*imgx  # 0-255 image
        #lcnt = 0
        #pixels = np.float32(imgx.reshape(-1,1))
        #for p in pixels:
                #if p > 125:
                    #lcnt+=1
        #print('{} points with label {}'.format(lcnt,lab))
  
        ##############
        #
        # find and disiplay the blobs
        #
        #keypoints = detector.detect(imgx)
        
        #print('I found {} blobs'.format(len(keypoints)))
        
        #for k in keypoints:
            #print('Angle: ', k.angle, 'octave: ',k.octave, 'size: ',k.size)
    
        #im_with_keypoints = cv2.drawKeypoints(imgx, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        ###############
        #
        #  find and display contours
        #
        imgy = imgx.copy()
        conts = []
        #imgray=cv2.cvtColor(imgy,cv2.COLOR_BGR2GRAY)
        #x, conts, hier = cv2.findContours(imgray,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, conts, hier = cv2.findContours(imgy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        print('I got {} contours'.format(len(conts)))
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob
        s2 = nf.scale*nf.scale
        ar_min = int(7500/s2)   # for areas use s-squared
        ar_max = int(100000/s2)
        timg = img0.copy()
        ncont=0
        for c in conts:
            #
            #  Filtering
            #
            drawfl = True
            # by area
            ca = cv2.contourArea(c)
            if ca < ar_min or ca > ar_max:
                drawfl = False
            #
            #  does it seem to be a rectangle??
            #
            # 
            rect = cv2.minAreaRect(c)
            w = rect[1][0]
            h = rect[1][1]
            if w * h != 0:
                aspect_r = w/float(h)
                extent = float(cv2.contourArea(c))/float(h*w)
            else:
                aspect_r = 1000000000
                extent = 0
            if len(c) > 10:
                (x,y), (MA,ma), orientation = cv2.fitEllipse(c)
            else:
                orientation = 0.0
            # correct orientation to apparent orientation on screen
            orientation = 270 -orientation
            
            obb = cv2.boxPoints(rect)
            obbp = np.int0(obb)
            x,y,w,h = cv2.boundingRect(c) 


            amax = 0.25
            emin = 0.6
            
            amax = 0.4
            emin = 0.4
            if aspect_r > amax:
                drawfl = False
            if extent < emin: 
                drawfl = False
            
            #print(np.shape(obbp))
            #print('obb:', obbp)
            
            if drawfl:
                #
                # location
                #            
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #print (rect)
                print('Properties: ')
                print('aspect: {:6.2f}  ext: {:6.2f} orient: {:6.2f}'.format(aspect_r,extent, orientation))

                ncont+=1
                cv2.drawContours(timg, c, -1, (255,0,255),3)
                cv2.drawContours(timg, [obbp], -1, (0,255,0),3)
                ob2 = [[ val*nf.scale for val in p]  for p in obbp]
                ob2 = np.array(obbp)*nf.scale
                
                cv2.drawContours(img_orig, [ob2], -1, (0,255,0), 3)
                xt = nf.scale*cX
                yt = nf.scale*cY
                cv2.putText(img_orig, '{}'.format(lab), (xt,yt), nf.font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        print(' ...    {} contours remain'.format(ncont))
        nfound += ncont
        
        if True:
            #cv2.imshow("Raw",img1)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            title = "Contours Identified/Filtered by label: {}".format(lab)
            blobwin = 'blobwin'
            #cv2.imshow(blobwin, timg)
            cv2.imshow(title, img_orig)
            cv2.waitKey(500)
            #time.sleep(5)
            cv2.destroyWindow(title)
    
    print('{}, {} booktangles detected'.format(pic_filename, nfound) )

    cv2.imshow(title, img_orig)
    cv2.waitKey(-1)
