
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
import os
import PIL
import init


class ImageProcess:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_name = os.path.splitext(os.path.basename(img_path))[0]
        self.origin = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.module_origin = None
        self.module_thresh = None
        self.persp_img= None
        self.persp_full= None        
        self.center_of_module = None
        
        
    # segmentation and save the transformed module images
    def segment(self,save_paths):
        img = self.origin
        gray =  cv2.GaussianBlur(img,(3,3),0)    
        thresh, result = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        kernel = np.ones((11,11),np.uint8)
        open = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel)
        
        # detect connected components
        cc = cv2.connectedComponentsWithStats(open,8,cv2.CV_32S)
        
        stats = cc[2]
        ind = np.argsort(stats[:,-1]) # indices of all connected components
        max_components = []
        min_val = stats[ind[-2],-1]/1.8 # to be used to exclude background
        mod_num = 1 # number of modues inside the image
    
        for i in reversed(range(ind.size-1)):
            # check if touching the boarder: test left, right, up and down, and also min area
            if (stats[ind[i],0] != 0 and stats[ind[i],-1] >= min_val and 
                (stats[ind[i],0]+stats[ind[i],2]) != img.shape[1] and
                (stats[ind[i],1] != 0) and
                (stats[ind[i],1]+stats[ind[i],3]) != img.shape[0]):
                #max_components.append((cc[1]==ind[i])*img)
                self.module_thresh = np.array((cc[1]==ind[i])*255, dtype=np.uint8)
                self.module_origin = (cc[1]==ind[i])*img
                self.center_of_module = np.flip(cc[3][ind[i],:],0)
                
                corners = self.corner_detection()
                self.perspective_transform(corners, init.params['dims'])
                                
                cv2.imwrite(save_paths[0]+str(mod_num)+'.jpg', self.module_origin)
                cv2.imwrite(save_paths[1]+str(mod_num)+'.jpg', self.persp_full) # save the perspective image
                mod_num += 1       


    # detect the four corner points
    # input: filename - path of the image file
    def corner_detection(self):  
        img = self.module_thresh
        _, contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # assume there is only one rectangle
        for cnt in contours:            
            rect = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if (len(rect) == 4):
                #arr = approx.reshape(4,2)
                diff = abs(np.max(rect, axis=0) - np.min(rect, axis=0))
                if (diff.sum() >= 200):
                    return self.computeCorners(rect)
                    
    # given four points with any order, find the right order
    # input: rect - numpy array containg four points           
    def computeCorners(self, rect):
        points = rect.reshape((4,2))
        ind = np.argsort(points[:,1])
        upper = points[ind,:][:2,:] # upper corner points
        lower = points[ind,:][2:,:] # lower corner points
        
        corners = np.zeros((4,2)) # the clockwise order: upper-left, upper-right, lower-right, lower-left
        corners[:2,:] = upper
        corners[2:,:] = lower
        if (upper[0,0] > upper[1,0]):
            corners[[0,1]] = corners[[1,0]]
        if (lower[0,0] < lower[1,0]): # !! from right to left
            corners[[2,3]] = corners[[3,2]]
        
        return corners 


    # perform perspective transformation   
    def perspective_transform(self, corners, dims):
        ratio = 1.0*dims[1]/dims[0]
        height = corners[:,1].max() - corners[:,1].min()
        corners = np.float32(corners)
        corners_new = np.float32([[0,0],[height*ratio,0],[height*ratio,height],[0,height]])
        corners_new += 50 # to make it sit in the middle of the image
        Proj = cv2.getPerspectiveTransform(corners, corners_new)
        dst = cv2.warpPerspective(self.module_origin, Proj, (int(height*ratio)+100,int(height)+100))
    
        self.persp_img = dst        
        self.perspective_centered()

    
    # save the perspectively transformed images as a whole image, not just a single module   
    def perspective_centered(self):
        self.persp_full = np.zeros_like(self.origin)
        x0, y0 = self.persp_full.shape
        x1, y1 = self.persp_img.shape
        
        # Get the left and upper corner of the module in the transformed image by
        # checking if touching left, upper, right, down
        center = self.center_of_module
        top_left = [max(center[0]-x1/2., 0), max(center[1]-y1/2., 0)]
        top_left = [min(top_left[0], x0-1-x1), min(top_left[1], y0-1-y1)]
        top_left = np.array(top_left).astype(int)
        
        self.persp_full[top_left[0]:top_left[0]+x1, top_left[1]:top_left[1]+y1] = self.persp_img.copy()
        
           

            
            
            
            