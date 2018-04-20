'''
Created on Feb 22, 2018

@author: jingpeng
'''

import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt

work_dir = '/home/jingpeng/work_dir/'


def motion_estimation(img1, imgk):
    rows, cols = img1.shape
    F = imgk - img1
    gx, gy = np.gradient(img1)
    
    X = np.tile(np.arange(rows)+1, (cols,1)).T
    Y = np.tile(np.arange(cols)+1, (rows,1))
    gxy = X*gy - Y*gx
    
    G = np.concatenate((gx,gy,gxy), axis=1)
    R = np.matmul(np.linalg.inv(np.matmul(G.T, G)), np.matmul(G.T,F))
    
    print(R)
def template_matching(template, img):
    h,w  = template.shape
    res= cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    
    # matching for only one object
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    print(res.max())
    
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    plt.show()

i =0
for i in range(3):
    print(i)
for i in range(8,11):
    print(i)
print(i)


#img1 = cv2.imread(work_dir+'2.jpg', 0) # 0.97879744
#img2 = cv2.imread(work_dir+'3.jpg', 0) # 0.9791671
#template = cv2.imread(work_dir+'template.jpg', 0)
#many_modules = cv2.imread(work_dir+'many_modules.jpg', 0)

#motion_estimation(img1, img2)
#template_matching(template, img2)
