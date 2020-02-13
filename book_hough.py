import cv2
import numpy as np
import glob as gb

import newfcns as nf

#scale = nf.scale
#scale = 3
#img_width = int(3000/scale)
#img_height =  int(4000/scale)
#img_width = int(1671)
#img_height =  int(1206)


# --------------Hough Transfrom---------
# 图像预处理（缩放、高斯滤波）
def pre_process(name):
    img = cv2.imread(name, 0)
    #img_resized = cv2.resize(img, (img_width, img_height))
    img_resized = img.copy()
    b = nf.blur_rad
    img_blur = cv2.GaussianBlur(img_resized, (b,b), 0)
    return img_resized, img_blur


def draw_hlines(img, lines):
    new_img = img.copy()
    for rho, theta in lines[:]: 
        x1,y1,x2,y2, m, b = rth2xymb(rho,theta)
        cv2.line(new_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return new_img

#
#  convert rho/theta to xy line 
#
def rth2xymb(rho,th):
    a = np.cos(th)
    b = np.sin(th)
    x0 = a*rho
    y0 = b*rho
    l = 4000 # big pixel value compared to image size 
    x1 = int(x0+(l*(-b)))
    y1 = int(y0+(l*a))
    x2 = int(x0-(l*(-b)))
    y2 = int(y0-l*a)
    if abs(x2-x1) < 0.00001:
        m = 10^6
    else:
        m = (y2-y1)/(x2-x1)
    b = y2-m*x2
    return (x1,y1,x2,y2, m,b)

def draw_vertical(img, lines):
    new_img = img.copy()
    for rho, theta in lines[:]:
        print ('drawing: ', rho, theta)
        x1 = rho
        x2 = rho
        y1 = 0
        y2 = img_height
        cv2.line(new_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return new_img

def draw_horizontal(img, y_val):
    y_val = int(y_val)
    new_img = img.copy()
    print('Drawing horizontal line at {}'.format(y_val))
    cv2.line(new_img, (0, y_val), (img_width, y_val), (255,0,0), 2)
    return new_img

def line_sample(n,lines):
    i = 0
    l2 = []
    for l in lines:
        if i%n==0:
           l2.append(l)
        i+=1
    return l2

def line_reduce(lines, y_frac,  dx_thresh):
    lines.sort()
    i = 0
    j = 0
    lines_final = []
    yval = int(img_height * y_frac)  # eg 25% of way up into the image 
    print('image dims: W:{} H:{}'.format(img_width, img_height))
    print(' dx_threshod: {}'.format(dx_thresh))
    while i < len(lines) - 1:
        if j >= len(lines) - 1:
            break
        j = i + 1
        lines_final.append(lines[i])
        x1, y1, x2dummy,y2dummy, m1, b1 = rth2xymb(lines[i][0],lines[i][1])
        x1a = (yval-b1)/m1  # x value of intersection with a horizontal line at 25% image height 
        while j < len(lines) - 1:
            rho = lines[j][0]
            theta = lines[j][1]
            # transform to cartesian lines so that we can get better clustering in x
            x1, y1, x2,y2, m2, b2 = rth2xymb(rho,theta)
            x1b = (yval-b2)/m2
            print('I got a line pair with xs {:8.2f} {:8.2f} and DX {:6.2f}'.format(x1a,x1b,x1a-x1b))
            if abs(x1a-x1b) > dx_thresh:  # is this line far enough in x from previous line?
                i = j
                break     # go back and add this line
            else:
                j = j + 1 # ignore and keep going
    return lines_final


def line_sifting(lines_list):
    lines = []
    
    #
    #   angle window relative to vertical 
    #
    window = 5 # +/- this many degrees
    wrad = window*np.pi/180.0  # window in rad
    ymax = np.sin(wrad)
    i = 0
    for rho, theta in lines_list[:]:
        thd = 360.0*theta/(2*np.pi)
        if rho < 0:
            rho *= -1
            theta += np.pi
        i+=1
        if i< 10:
            print('  {:5}:  rho: {:7.0f} th: {:4.2f}(deg)'.format(i,rho,thd))
        ytmp = np.sin(theta)
        g1 = (ytmp <= ymax) and (ytmp >= -ymax)      # 3:00 or 9:00 +/- window
        if (g1):  # filter the line angles
            lines.append([rho, theta])
            print('>>{:5}:  rho: {:7.0f} th: {:4.2f}(deg)'.format(i,rho,thd))

    print('Sifting: I got ', np.shape(lines), ' lines')
    return lines


# ------------Region Grow---------------
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


def get_seeds(lines):
    seeds = []
    i = 0
    j = 1
    while i < len(lines)-2:
        y = int(lines[i][0] + (lines[j][0] - lines[i][0])/2) # 图片索引的x、y与我们理解的x、y相反
        x = int(img_height/2)
        seeds.append(Point(x, y))
        i = i + 1
        j = j + 1
    return seeds


def get_gray_diff(img, current_point, adjacent_point):
    return abs(int(img[current_point.x][current_point.y]) - int(img[adjacent_point.x][adjacent_point.y]))


def get_connects():
    connects = [Point(-1, -1), Point(-1, 0), Point(-1, 1), Point(0, -1), Point(0, 1), Point(1, -1), Point(1, 0),
                Point(1, 1)]
    return connects


def region_grow(img, seeds, thresh):
    seed_mark = np.zeros(img.shape)
    seed_stack = []
    for seed in seeds:
        seed_stack.append(seed)
    mark = 1
    connects = get_connects()
    while len(seed_stack) > 0:
        current_point = seed_stack.pop(0)
        seed_mark[current_point.x][current_point.y] = mark
        for connect in connects:
            adjacent_x = int(current_point.x + connect.x)
            adjacent_y = int(current_point.y + connect.y)
            if adjacent_x < 0 or adjacent_y < 0 or adjacent_x >= img_height or adjacent_y >= img_width:
                continue
            gray_diff = get_gray_diff(img, current_point, Point(adjacent_x, adjacent_y))
            if gray_diff < thresh and seed_mark[adjacent_x][adjacent_y] == 0:
                seed_mark[adjacent_x][adjacent_y] = mark
                seed_stack.append(Point(adjacent_x, adjacent_y))
    return seed_mark



# --------------image segmentation---------------
def segmentation(img, lines): 
    imgs = []
    bounding_lines = []
    i = 0
    j = 1
    book_min_width = 69
    while i < len(lines) - 2:
        x1 = int(lines[i][0])
        x2 = int(lines[j][0])
        if abs(x1-x2) > book_min_width:
            book_img = img[0:img_height, x1:x2]
            imgs.append(book_img)
            bounding_lines.append(lines[i])
        i = i + 1
        j = j + 1
    print ('I found {} books'.format(len(imgs)))
    return imgs, bounding_lines


def seg_horizontal(img):
    thresh = img.shape[1] - 10
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines_pre = cv2.HoughLines(edges, 1, np.pi / 180, thresh)  # 最后一个参数可调节，会影响直线检测的效果
    lines = lines_pre[:, 0, :]
    lines_horizontal = []
    for rho, theta in lines[:]:
        if ((theta < (12 * np.pi / 18.0)) and (theta > (4 * np.pi / 18.0))) or ((theta > (22 * np.pi / 18.0)) and (theta < (32 * np.pi / 18.0))):
            lines_horizontal.append([rho, theta])
    lines_horizontal.sort()
    lines_horizontal = line_reduce(lines_horizontal)
    if len(lines_horizontal) == 0:
        return img
    y1 = int(lines_horizontal[0][0])
    y2 = int(lines_horizontal[len(lines_horizontal)-1][0])
    book_img = img[y1:y2, 0:img_width]
    return book_img


