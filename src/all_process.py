
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
from setup import *
from matching import segment_modules









# transform all the single module images 
def perspective_all(dir_module, only_for_center):
    # for all the modules iamges, compute the perspective transform
    for filename in sorted(os.listdir(dir_module), key=lambda x: (int(x.split('_')[0]), int(x.split('.')[0].split('_')[1]))):
        video = video_process(dir_module+filename)
        video.segment(video.origin_img)
        
        corners = video.houghLine(video.module_thresh)
        video.perspective(video.origin_img, corners, dims)
        #cv2.imwrite(match_persp+os.path.splitext(filename)[0] + '.jpg', video.persp_img)
        
        # save the centroids of these modules for later use, i.e. 
        # acting as a reference point for transformation
        center = video.center_of_module
                
        # save the perspectively transformed images as a whole image, not just a single module    
        img = np.zeros_like(video.origin_img)
        x0, y0 = img.shape
        x1, y1 = video.persp_img.shape
        
        # Get the left and upper corner of the module in the transformed image by
        # checking if touching left, upper, right, down
        top_left = [max(center[0]-x1/2., 0), max(center[1]-y1/2., 0)]
        top_left = [min(top_left[0], x0-1-x1), min(top_left[1], y0-1-y1)]
        top_left = np.array(top_left).astype(int)
        
        img[top_left[0]:top_left[0]+x1, top_left[1]:top_left[1]+y1] = video.persp_img.copy()        
        
        cv2.imwrite(match_persp_full + os.path.splitext(filename)[0] + '.jpg', img)
        
    
            

            