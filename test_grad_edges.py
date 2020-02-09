
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


def drawv2(img, xvals):
    for x in xvals:
        x1 = x
        x2 = x
        y1 = 0
        y2 = img_height
        cv2.line(img, (x1,y1),(x2,y2), (255,255,255),2)
        

l1 = []
f = open('ctrans_labels.csv','r')
for row in f:
    y = row.split(',')[1]
    l1.append(float(y))
   
line = np.array(l1) 

edge_x = nf.Find_edges_line(line)
print('{} edge x vals:'.format(len(edge_x)))
print (edge_x)

#img_gray, img = pre_process(pic_filename)
#cv2.IMREAD_GRAYSCALE

#
#  read in and blur the image
#
pic_filename = "tiny/topshelf001.jpg"
img1 = cv2.imread(pic_filename, cv2.IMREAD_COLOR)

#cv2.imshow("Image", img1)
 
print('image read in')
drawv2(img1, edge_x)

print(type(img1))

cv2.imshow("New Book Boundaries", img1)


cv2.waitKey()
cv2.destroyAllWindows()
