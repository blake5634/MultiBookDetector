
import cv2
import numpy as np
import time as time
import glob as gb
from book_hough import *
import newfcns as nf
import matplotlib.pyplot as plt
'''
Test some new functions for drawing in mm on the image
'''


pic_filename = 'tiny/target.jpg'
#
#img = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
img_orig = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
ish = img_orig.shape

img = img_orig.copy()

# draw a rectangle (-20mm, -20mm) --> (20mm,20mm)
nf.DLine_mm(img, (-20,-20), (20,20), 'white', iscale=1)
nf.DRect_mm(img, (-20,-20), (20,20), 'red', width=3,iscale=1)

nf.DRect_mm(img, (-100,-100),(100,100), 'green',width=-1,scale=1)

title='test image'
cv2.imshow(title, img)
cv2.waitKey(3000)
