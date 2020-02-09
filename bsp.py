import cv2
import numpy as np
import glob as gb
from book_hough import *
import newfcns as nf
'''
Find books in a bookshelf image


'''
scale = 3
img_width = int(3000/scale)
img_height =  int(4000/scale)
img_width = int(1671)
img_height =  int(1206)


def maincode():
    SHOW_PIPE = True
    img_paths = gb.glob("tiny/*.jpg")
    if (len(img_paths) < 1):
        print('No files found')
        quit()
    for pic_filename in img_paths:
        print('looking at '+pic_filename)
        img_gray, img = pre_process(pic_filename)
        
        if False:
            cv2.imshow("gray",img_gray)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        #
        # stat-based thresholds
        v = np.median(img_gray)
        sigmal = 0.4  # parameter
        sigmah = 0.33
        #---- apply automatic Canny edge detection using the computed median----
        lower = int(max(  0, (1.0 - sigmal) * v))
        upper = int(min(255, (1.0 + sigmah) * v))
        
        print ('lower: {} vs.  50'.format(lower))
        print ('upper: {} vs. 150'.format(upper))
        
        #edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        edges = cv2.Canny(img_gray, lower,upper, apertureSize=3)
        
        if False:
            cv2.imshow("Canny",edges)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        houghthreshold = 275
        lines = cv2.HoughLines(edges, 1, np.pi/180, houghthreshold)  # 最后一个参数可调节，会影响直线检测的效果
        if type(lines) == type(None):
            print('No lines were found')
            quit()
        elif len(lines) < 5:
            print('Not enough lines')
            quit()
                
        lines1 = lines[:, 0, :]  # just get rho, th pairs
        #fewlines = line_sample(1,lines1)
        fewlines = lines1
        #
        #  line sifting()
        #
        houghlines = line_sifting(fewlines)  # 存储并筛选检测出的垂直线 (store & filter detected vertical lines)
        img_show = draw_hlines(img_gray, houghlines)
        #img_show = draw_vertical(img_gray, houghlines)
        print('from {} lines to {} lines.'.format(len(lines1),len(houghlines)))
        
        if SHOW_PIPE:
            cv2.imshow("Sifted Hough Lines",img_show)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
             
        #
        #   line reduce() 
        #
        lineheight = 0.75
        l1 = len(houghlines)
        lines2 = line_reduce(houghlines, lineheight, 60)
        l2 = len(lines2)
        print('line reduce: I reduced from {} to {} lines'.format(l1,l2))        
        img_3 = draw_hlines(img_gray, lines2)
        
        if SHOW_PIPE:
            img_3 = draw_horizontal(img_3, lineheight*img_height)
            cv2.imshow("Reduced Hough Lines",img_3)
            cv2.waitKey()
            cv2.destroyAllWindows()
         
        img_segmentation, bound_lines = segmentation(img_gray, houghlines)
        img_book_bounds = draw_hlines(img_gray,bound_lines)
        
        if SHOW_PIPE:
            cv2.imshow("Book Bounding Lines",img_book_bounds)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        i = 0
        for img_s in img_segmentation:
            if img_s.shape[0] == 0:
                print(i)
            # img_s = seg_horizontal(img_s)
            str1 = pic_filename[6:]
            str1 = str1[:-4]
            string = 'results/' + str1 + "-" + str(i) + '.jpg'
            print("Write " + string)
            #cv2.imwrite(string, img_s)  # 保持切割后的图像
            i = i+1 

if __name__ == "__main__":
    maincode()
    
    
