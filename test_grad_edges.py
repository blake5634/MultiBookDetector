
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


def draw_gaps(img, gaps, gsc=1):
    i=0
    for g in gaps:
        i+=1
        x1 = g[0]*gsc
        x2 = g[1]*gsc
        col = (0,255,0)

        y1 = 0
        y2 = img_height*gsc
        y3 = int(gsc*img_height/2)+10*(i%5)
        cv2.line(img, (x1,y1),(x1,y2), col ,2)
        cv2.line(img, (x2,y1),(x2,y2), col ,2)
        cv2.line(img, (x1,y3),(x2,y3), col ,8)
#

def drawv3(img, cross):
    for c in cross:
        if c < 0: # start of a book
            col = (255,255,255)
            c = -c
        else:
            col = (0,255,0)
        x1 = c
        x2 = c
        y1 = 0
        y2 = img_height
        cv2.line(img, (x1,y1),(x2,y2), col ,2)
        
#


def drawv2(img, xvals):
    for x in xvals:
        x1 = x
        x2 = x
        y1 = 0
        y2 = img_height
        cv2.line(img, (x1,y1),(x2,y2), (255,255,255),2)
        
#
#
#    start of test
##

#
#  read in the image
#
pic_filename='tiny/topshelf001.jpg'

pic_filename='tiny/target.jpg'

img = cv2.imread(pic_filename, cv2.IMREAD_COLOR)
ish = img.shape
#
#  scale the image 
#
#     scale factor imported from newfcns.py
#
    
img_width = int(ish[1]/nf.scale)
img_height =  int(ish[0]/nf.scale)

####################
# 
# save time - read line of labels (K-means)
#
l1 = []
f = open('ctrans_labels.csv','r')
for row in f:
    y = row.split(',')[1]
    l1.append(float(y))
line = np.array(l1) 

f = open('metadata.txt','r')
for l in f:
    txt, val = l.split(',') 
f.close()
black_label=int(val)

#
#   plot a graph of original and gradient
x = range(len(line))
fig, ax = plt.subplots()
#print(':::: sline: ',type(sline), np.shape(sline))
ax = plt.plot(x,line)


#
#  Get derivative of labels
#
#   deriv_win_size in newfcns
# 
win = int(nf.deriv_win_size/nf.scale)
if win%2==0:
    win+= 1
for w in [win]:
    print('estimating derivative of {} labels using {} window'.format(len(line),w))
    grad = nf.Est_derivative(line, w)
    gs = []
    for g in grad:
        gs.append( g)
    ax = plt.plot(x,gs)
 
 
#
#  Get the Zero Crossings of label deriv. 
#
cross = nf.Find_crossings(gs)


#
#   Plot + and - edges on curve
#
no = -2
yes = 8
xm = []
ym = []
x0 = 0
y0 = no
for c in cross: 
    if c < 0:     ##  start
        c = -c
        y1 = yes
    else:
        y1 = no
        
    x1 = c
    x2 = c
    xm.extend([x0, x1, x2])
    ym.extend([y0, y0, y1])
    x0 = x1
    y0 = y1
    
ax = plt.plot(xm,ym) 
plt.title('line and its gradients')
plt.grid(True)
plt.show()
 

#
#   study gaps btwn edges for likely books
#
#

#  identify black (background)
#      and skinny gaps
#  "6" = black

gaplist = nf.Gen_gaplist(cross)
print('There are {} gaps'.format(len(gaplist)))
print (gaplist)
cands = nf.Gap_filter(gaplist, img, tmin=int(40/nf.scale), blacklabel=black_label)
print('After filtering:')
print('There are {} gaps left'.format(len(cands)))
############################################################
#
#    Display book boundaries on image
#
draw_gaps(img,cands,nf.scale)
 
cv2.imshow("New Book Boundaries", img)

cv2.waitKey()
cv2.destroyAllWindows()
