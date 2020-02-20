import cv2
import numpy as np
import glob as gb
import matplotlib.pyplot as plt

scale = 3   # downsample image by this much

#   ~5.0 pix / mm  (measured from original target img)

pix2mm = float(.2) * float(scale)   # convert pix * pix2mm = xx mm
                                   # measure off unscaled target image
mm2pix = float(1.0)/pix2mm 

#  following smoothing windows are scaled from here by /scale
#     values below reflect a nominal image width of 1670

deriv_win_size =     int(1.0*mm2pix)      # 1mm width
smooth_size    = -1* int(10*mm2pix)    # <0:   do not smooth
blur_rad       =     int(1.0*mm2pix)    # to scaled pixels
if blur_rad%2 == 0:   # make it odd # of pixels
    blur_rad += 1   

KM_Clusters = 10

font = cv2.FONT_HERSHEY_SIMPLEX
colors = {'white':(255,255,255), 'blue':(255,0,0), 'green':(0,255,0), 'red':(0,0,255)}
 
#
#  new functions by BH
#
##


#
# Convert XY rel LL corner to row, col
#
##  from   XY referenced to center of img:
#
#                       |Y
#                       |
#                       |
#      -----------------------------------
#         X             |\
#                       | (ih/2, iw/2)
#                       |
#                       |
#
#    to classic image proc coordinates:
#      0/0------- col ----->
##      |
#       row
#       |
#       |
#       V
##
def Get_pix_byXY(img,X,Y):
    #print('Get: {} {}'.format(X,Y))
    row,col = XY2RC(img,X,Y)
    if col > 500:
        print('  Get_pix_byXY() X:{} Y:{} r:{} c:{}'.format(X,Y,row,col))
    return(img[row,col])

def Get_pix_byRC(img,row,col): 
    return(img[row,col])

#
# convert image ctr XY(mm) to X,Y (open CV point)
#
def XY2iXiY(img,X,Y,iscale=scale):
    if iscale != scale:   # mm2pix has scale in it so..
        f = float(scale)/float(iscale)
        X *= f
        Y *= f
    row = int( -Y*mm2pix + int(img.shape[0]/2) )
    col = int(  X*mm2pix + int(img.shape[1]/2) )
    iX = col
    iY = row
    return iX, iY
#
# convert image ctr XY(mm) to Row, Col 
#
def XY2RC(img,X,Y,iscale=scale):
    if iscale != scale:
        f = float(scale)/float(iscale)
        X *= f
        Y *= f
    row = int( -Y*mm2pix + int(img.shape[0]/2) )
    col = int(  X*mm2pix + int(img.shape[1]/2) )
    return row,col

#
#  Get image bounds in mm  
#
def Get_mmBounds(img,iscale=scale):
    sh = np.shape(img)  # get rows & cols
    xmin = -1* (sh[1]*pix2mm/2)  # pix2mm factor includes scale
    xmax = -1*xmin
    ymin = -1* (sh[0]*pix2mm/2)
    ymax = -1*ymin
    if iscale != scale:
        f = float(iscale)/float(scale)
        xmin *= f
        xmax *= f
        ymin *= f
        ymax *= f
    return (xmin, xmax, ymin, ymax)
#
#  Draw a line/rect in mm coordinates
#
#  if image scale is different from "scale" then use param
#
def DLine_mm(img, p1, p2, st_color, width=3,iscale=scale):
    p1_pix = XY2iXiY(img, p1[0],p1[1],iscale=iscale)
    p2_pix = XY2iXiY(img, p2[0],p2[1],iscale=iscale)    # allows for change of scale 
    cv2.line(img, p1_pix, p2_pix, colors[st_color], width)
    
def DRect_mm(img,  p1, p2, st_color, width=3,iscale=scale):
    p1_pix = XY2iXiY(img, p1[0],p1[1],iscale=iscale)
    p2_pix = XY2iXiY(img, p2[0],p2[1],iscale=iscale)
    cv2.rectangle(img, p1_pix, p2_pix, colors[st_color], width)
 
#
#  Return standard image size references
#    img = unscaled image
#    scale = int scale factor to be used
def Get_sizes(img, scale):
    ish = np.shape(img)
    siw = int(ish[1]/scale)
    sih =  int(ish[0]/scale)
    return ish, (sih, siw)


#
#
#  Get edge score of a line through image
#
#   y = mx+b  (y=row, x=col)
#
#   NEW:   All coordinates and radii etc are in mm 
def Get_line_score(img, w, xintercept, th, llen,bias, cdist):
    '''
    img = image (already scaled)
    w   = width of line analysis window (90deg from line) (mm)
    xintercept = where line crosses vertical centerline of image (X=0) (mm)
    th  = angle in deg relative to 03:00 on clock
    llen = length of line segment (mm)
    cdist = matrix of color distances (Euclid) btwn VQ centers
    '''
    print('\n\n w: {} xint: {}, th: {}'.format( w,xintercept, th))
    print(' ---   image shape: {}'.format(np.shape(img)))
    print('Image sample: {}'.format(img[10,10]))
    ih = img.shape[0]
    iw = img.shape[1] 
    xmin, xmax, ymin, ymax = Get_mmBounds(img)  # in mm
    assert (xintercept > xmin and xintercept <= xmax), 'bad x-value: '+str(xintercept)
    d2r = 2*np.pi/360.0  #deg to rad
    m0 = np.tan(th*d2r) # slope (mm/mm)
    b0 = -m0*xintercept  # mm
    #bp, dummy = XY2iXiY(img,b0,0)  # pixels (cols)
    #window upper bound
    #  rV = distance to upper bound (vertical) mm
    
    rV  = abs( w/np.cos((180-th)*d2r)) # mm  (a delta / no origin offset)
    rVp = int(rV * mm2pix)     # pixels
    print('m0: {} b0: {}mm rVp:{}(pix)'.format(m0,b0,rVp))
    print('rV(mm): {:5.2f}'.format(rV))
    print('th: {} deg, iw {}  ih: {}'.format(th,iw,ih))
    dx = abs((llen/2)*np.cos(th*d2r)) # mm
    
    xmin2 = xintercept - dx #mm    X range for test line
    xmax2 = xintercept + dx #mm
    print('xmin/max2: {:4.2f}mm {:4.2f}mm'.format(xmin2,xmax2))
    # cols,  rows = XY2iXiY()
    xmi2p, dummy =XY2iXiY(img, xmin2,0)  # pix  X range for test line
    xmx2p, dummy =XY2iXiY(img, xmax2,0)
    
    rng = range(xmi2p, xmx2p-1, 1)  # pix cols
    print('x range: {} -- {}'.format(xmi2p, xmx2p)) 
    #study pixels above and below line at all columns
    vals_abv = []
    vals_bel = []
    ymaxp = ih    # pix, same as image rows
    yminp = 0     # pix
    for col in rng:
        x = pix2mm*(col - iw/2) # convert back to mm(!)
        ymm = m0*x+b0     # line eqn in mm
        row, dummy = XY2RC(img,0,ymm)    # pix
        #print ('X:{} Y{}'.format(x,y),end='')
        if (row > ih-1 or row < 0) or (col > iw-1 or col < 0): # line inside image?
            #print('')  # no it's not inside
            continue
        else:
            #print('*')
            # above the line
            for row1 in range(row,row-rVp,-1): # higher rows "lower"
                if  row1 < ymaxp:
                    #print('             row range1: {} -- {}'.format(row,row+r))
                    vals_abv.append(Get_pix_byRC(img,row1,col)) # accum. labels in zone above
            # below the line
            for row1 in range(row, row+rVp,1):
                if row1 > yminp:
                    #print('             row range2: {} -- {}'.format(row,row-r))
                    vals_bel.append(Get_pix_byRC(img,row1,col))
    print('\n\n{} values above'.format(len(vals_abv)))
    print('{} values below'.format(len(vals_bel)))
    if len(vals_abv) > 50 and len(vals_bel) > 50:
        #print('shape vals: {}'.format(np.shape(vals_abv)))
        #print('sample: vals: ', vals_abv[0:10])
        labs_abv, cnts_abv = np.unique(vals_abv, return_counts=True)
        labs_bel, cnts_bel = np.unique(vals_bel, return_counts=True)
        print('shape: labels_abv: {}, counts_abv: {}   Data: '.format(np.shape(labs_abv),np.shape(cnts_abv)))
        print(labs_abv, cnts_abv)
        print('labels: (100 samples)')
        print(vals_abv[0:100])
        dom_abv = np.max(cnts_abv)/np.sum(cnts_abv)  # how predominant? (0-1)
        dom_bel = np.max(cnts_bel)/np.sum(cnts_bel)  # how predominant? (0-1)
        color_distance = cdist[np.argmax(cnts_abv),np.argmax(cnts_bel)]
        cl_abv = labs_abv[np.argmax(cnts_abv)] # most common above
        cl_bel = labs_bel[np.argmax(cnts_bel)] # most common below
        diff_score = (color_distance)*dom_abv*dom_bel  # weighted difference
        #diff_score = dom_abv*dom_bel
        print('color cluster diff: {:8.3f}'.format(color_distance))
        print('cl_abv/bel: {}/{} dom_abv/bel: {:5.3f}/{:5.3f}, score:  {}'.format(cl_abv,cl_bel,dom_abv,dom_bel,diff_score))
        #x = input('pause ...')
    else:
        return 0.0
    return diff_score
                    
                
            
#
#  Check along top for typical bacgkround pixels
#

##########################
#
#   Look across top of image for label of "black"
#
def Check_background(lab_image, outfile=False):
    bls = []
    wheight = int(lab_image.shape[0]/8)
    for col in range(lab_image.shape[1]):
        for r in range(wheight):
            bls.append(lab_image[r+5,col])
    labs, cnt = np.unique(bls, return_counts=True)
    max = np.max(cnt)
    for l,c in zip(labs,cnt):
        if c == max:
            lmax = l
            break
    print('Most common label near top: {} ({}%)'.format(lmax, 100*max/np.sum(cnt)))
    if outfile:
        f = open('metadata.txt','w')
        print('dark label, {}'.format(lmax), file=f)
        f.close()
    return lmax


#
#  Apply criteria to gaps
#
def Gap_filter(gaps,img, tmin=20, blacklabel=6):
    img_height=img.shape[0]
    img_width=img.shape[1]
    # tmin:        min width pixels
    # blacklabel:  int label of background black

    halfway = int(img_height/2)
    candidates = []
    for g in gaps:
        #
        #  exclude
        #
        # 1) narrow gaps
        width = abs(g[0]-g[1])
        if width < tmin:
            print('found a very narrow gap')
            continue
        # 2) gaps that match background
        values = []
        for c1 in range(width):
            col = g[0]+c1-1
            for r in range(halfway):
                row = halfway + r -1 
                if col < img_width:
                    values.append(img[row,col])
        (val,cnts) = np.unique(values, return_counts=True)
        if val[np.argmax(cnts)] == blacklabel:  # background
            print('found a black gap')
            continue
        else: # we didn't exclude this gap
            candidates.append(g)
    return candidates
#
#
#
def Gen_gaplist(cross): 
    cm1=0
    gaps = []
    for c in cross:
        if c < 0:
            c = c*-1
        gaps.append([cm1,c])
        cm1 = c
    return gaps

#
#  neg and pos zero crossings
#

def Find_crossings(yvals):
    ym1 = yvals[0]
    c = []
    for i,y in enumerate(yvals):
        if y<0 and ym1>=0:
            c.append(-i)   # - == neg crossing
        if y>0 and ym1 <= 0:
            c.append(i)    #   positive crossing
        ym1 = y
    return c
            

def Est_derivative(yvals, w):
    if w < 0:
        return np.gradient(line,2)
    if w > len(yvals)/2:
        print(' derivative window {} is too big for {} values.'.format(w,len(yvals)))
        quit()
    else:
        ym1 = yvals[0]
        dydn = []
        dn = 1 # for now
        for y in yvals:
            dy = y-ym1
            dydn.append(dy/dn)
            ym1 = y
        if w>1:
            dydn = smooth(dydn, window_len=w, window='hanning')        
        return dydn
 
    
def Find_edges_line(line):
    
    #grad = np.gradient(line,2)
    
    grad = Est_derivative(line, 3)
        
    thresh = 0.1
    edges = []
    for i,gv in enumerate(grad):
        if abs(gv) > thresh:
            edges.append(i)

    # get "edge of edges"
    e2 = []
    ep = edges[0]
    for e in edges:
        if e-ep > 1:    # leading edge of each gradient peak
            e2.append(e)
        ep = e
    edges = e2
    
    
        
    return edges

#
#   select a horizontal strip
#    return a series of labels
# 

def Trancept_labeled(lab_img, yval):
    img_height=img.shape[0]
    img_width=img.shape[1]    
    r = int(yval)
    result = []
    for c in range(img_width):
        lab_pix = lab_img[r,c]
        result.append(lab_pix)
    return result
#
#   Same as Trancept but return most common 
#     label in a vertical bar of width bw
#
#  cluster mean = avg value of pixel labels

def Trancept_bar_labeled(lab_img, yval,bw):    
    img_height=lab_img.shape[0]
    img_width=lab_img.shape[1]
    y_val = int(yval) # y=row, x=col
    result = []
    offset = int(bw/2)
    vv_array = []
    for x in range(img_width):
        vertvals = []
        for i in range(bw):
            y = y_val-offset + i
            if y < img_height:
                vertvals.append(lab_img[y,x])
        if len(vertvals) > 1:
            (val,cnts) = np.unique(vertvals,return_counts=True)
            result.append(val[np.argmax(cnts)]) # return most common label in the vert bar
        else:
            result.append(-2)
        vv_array.append(vertvals)
    return result, vv_array


#
#  generate an image illustrating the KM cluster center colors
#
def Gen_cluster_colors(centers):
    FILLED = -1
    ih = 600
    iw = 300
    img = np.zeros((ih,iw,3), np.uint8)
    h = int(ih/10)
    y=0
    x=0
    for i in range(len(centers)):
        col = tuple([int(x) for x in centers[i]])
        print('Color label {}: '.format(i),col)
        if i >= 10:
            break
        cv2.rectangle(img, (x,y), (iw,y+h), col, FILLED)
        cv2.putText(img, 'cluster: {}'.format(i), (50, y+50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        y=y + h
    return img
#   
#  Cluster colors by K-means
#
def KM(img,N):
    img_height=img.shape[0]
    img_width=img.shape[1]
    pixels = np.float32(img.reshape(-1, 3))
    n_colors = N
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .05)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, (centers) = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
#And finally the dominant colour is the palette colour which occurs most frequently on the quantized image:

    dominant = centers[np.argmax(counts)]  # most common color label found
    
    labeled_image = labels.reshape(img.shape[:-1])
    #print('i  label (n pix)    Pallette')
    #for i in range(len(labels)):
        #print('{}   {} ({}) '.format(i,labels[i],counts[i]), palette[i])
     
    # compute a distance matrix between the cluster centers (color similarity)
    dist = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            dist[i,j] = cv2.norm(centers[i]-centers[j])
            
    # from float back to 8bit
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    newimg = centers[labels.flatten()]
    
    #reshape
    newimg = newimg.reshape(img.shape)
    
    return [newimg, labeled_image, centers, dist]


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    #assert x.ndim != 1, "smooth only accepts 1 dimension arrays."

    #print ('input data:', len(x))
    #print (x)
    assert (len(x) > window_len),"Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if window_len%2 == 0:
        print(' smoothing window length must be ODD')
        quit()
        
    assert (window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']), "Window must be: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y to original length
    extra = len(y)-len(x)
    endchop = int(extra/2)
    print('extra: {}, endchop: {}'.format(extra, endchop))
    z = y[endchop:-endchop]
    print('Orig len: {}  New len: {}'.format(len(x), len(z)))
    return z

