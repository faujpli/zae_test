import numpy as np
import cv2
import os.path
from matplotlib import pyplot as plt
from processing import *
from PIL import Image
import SR.SR as sr
import scipy.misc
from setup import *


def detection(fn):
    fn = img_dir + fn
    img = cv2.imread(fn) # cv2.IMREAD_GRAYSCALE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) 
    
    '''  display the image '''
    if not os.path.exists(fn):
        print('No such image!')
        return
    # cv2.imshow('EL-image', img)
    # cv2.waitKey(0)
    
  
    
     # option 1: Horris corner detection
    

    dst = cv2.cornerHarris(gray, 5, 7, 0.01)
    dst = cv2.dilate(dst, None)
    
    img[dst>0.01*dst.max()] = [0, 0, 255]
    cv2.imshow('dst', img)
    cv2.waitKey(60000) 
    
    ''' option 2: Contour '''

    # x,y,w,h = cv2.boundingRect(img)
    
    ''' option 3: detect straight lines '''
    
    
    print("pass")


# input: grayscale image
# output: segmented image
def segment1(img):
    # first threasholding
    #ret, thresh = cv2.threshold(img,10,230,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #kernel = np.ones((3,3),np.unit8)
    #open = cv2.morphologyEx(img, cv2.MORPH_OPEN,kernel,iterations=2)
    pass




file = work_dir + '3_module.jpg'
img = Image.open(file).convert('L')
img = np.asarray(img)
img= img.astype(np.float32) / 255.0

eps = 0.5
#guided_img1 = sr.guided_Filter(img, img, 1, eps)
#guided_img3 = sr.guided_Filter(img, img, 3, eps)


img1 = cv2.bilateralFilter(img,8, 10, 0.5)
scipy.misc.imsave(work_dir + 'results/filtered.jpg', img1)
#scipy.misc.imsave(work_dir + 'filtered.jpg', img1)

# image unshap -- does not work well!
#gaussian = cv2.GaussianBlur(img, (3,3), 0.1)
#unsharp = cv2.addWeighted(img, 2, gaussian, -0.9, 0)

#plt.imshow(unsharp, 'gray')
#scipy.misc.imsave(work_dir + 'results/filtered.jpg', unsharp)
plt.show() 

print("finished")








