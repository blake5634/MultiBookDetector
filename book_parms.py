import cv2

scale = 2  # downsample image by this much

#   ~5.0 pix / mm  (measured from original target img)

pix2mm = float(.2) * float(scale)   # convert pix * pix2mm = xx mm
                                   # measure off unscaled target image
mm2pix = float(1.0)/pix2mm 

#  following smoothing windows are scaled from here by /scale
#     values below reflect a nominal image width of 1670

deriv_win_size =     int(1.0*mm2pix)      # 1mm width
smooth_size    = -1* int(10*mm2pix)    # <0:   do not smooth
blur_rad       =     int(7.0*mm2pix)    # to scaled pixels
if blur_rad%2 == 0:   # make it odd # of pixels
    blur_rad += 1   

KM_Clusters = 14

#
#   slant.py
#
Line_Score_Thresh = 20  # score units
max_gap_mm = 6  #mm   gap btwn lines defining new cluster (slant.py)
book_edge_line_length_mm = 120 # mm  line length checked for edges
slice_width  = 5 # mm   width examined for edges on either side of line
line_bias = -25  # mm   (shift line down from Y=0 line)         

#  method 1 == Euclidean RGB distance
#  method 2 == 350 if labels different, 0 otherwise
Color_Dist_Method = 2

font = cv2.FONT_HERSHEY_SIMPLEX
colors = {'white':(255,255,255), 'blue':(255,0,0), 'green':(0,255,0), 'red':(0,0,255),'yellow':(0,255,255)}
 
