
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
from setup import *
from matching import segment_modules



class ImageProcess:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_name = os.path.splitext(os.path.basename(img_path))[0]
        self.origin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.thresh = None
        self.module_origin = None
        self.module_thresh = None
        self.contour = None
        self.persp_img= None
        self.cell_img = None
        self.center_of_module = None
        
        
    
    def segment(self): #,save_path):
        img = self.origin_img
        gray =  cv2.GaussianBlur(img,(3,3),0)    
        thresh, result = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        kernel = np.ones((11,11),np.uint8)
        open = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel)
        
        # detect connected components
        cc = cv2.connectedComponentsWithStats(open,8,cv2.CV_32S)
        
        stats = cc[2]
        ind = np.argsort(stats[:,-1]) # indices of all connected components
        max_components = []
        min_val = stats[ind[-2],-1]/1.8 # exclude background
        mod_num = 1 # number of modues inside the image
    
        for i in reversed(range(ind.size-1)):
            # check if touching the boarder: test left, right, up and down, and also min area
            if (stats[ind[i],0] != 0 and stats[ind[i],-1] >= min_val and 
                (stats[ind[i],0]+stats[ind[i],2]) != img.shape[1] and
                (stats[ind[i],1] != 0) and
                (stats[ind[i],1]+stats[ind[i],3]) != img.shape[0]):
                #max_components.append((cc[1]==ind[i])*img)
                #cv2.imwrite(save_path+str(mod_num)+'.jpg', (cc[1]==ind[i])*img)
                mod_num += 1       


    # detect the four corner points
    # input: filename - path of the image file
    def corner_detection(self):  
        #_,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, contours, _ = cv2.findContours(self.thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
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
    def perspective(self, img, corners, dims):
        ratio = 1.0*dims[1]/dims[0] 
        height = corners[:,1].max() - corners[:,1].min()
        corners = np.float32(corners)
        corners_new = np.float32([[0,0],[height*ratio,0],[height*ratio,height],[0,height]])
        corners_new += 50 # to make it sit in the middle of the image
        Proj = cv2.getPerspectiveTransform(corners, corners_new)
        dst = cv2.warpPerspective(img, Proj, (int(height*ratio)+100,int(height)+100))
    
        self.persp_img = dst
        self.cell_img = self.persp_img.copy()
    

            
            
            
            