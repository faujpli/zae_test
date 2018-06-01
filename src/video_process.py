# basic steps: image acquisition -> image segmentation -> best image selection
# segmentation: obtain the image of individual modules
# 

# image quality measurement: find the best image out of the  module images
# 

# possibly: superresolution reconsturction; reshape the images to be rectangular 






import numpy as np
import math
import cv2
import os.path
from matplotlib import pyplot as plt
from scipy import stats
from scipy import signal 
#from scipy.stats.mstats_basic import signaltonoise
from multiprocessing import process
from skimage import feature
from statistics  import median_low
from setup import *



class video_process:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_name = os.path.splitext(os.path.basename(img_path))[0]
        self.origin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.thresh = None
        self.module_origin = None
        self.module_thresh = None
        self.hough = None
        self.persp_img= None
        self.cell_img = None
        self.center_of_module = None
        
                
    def extract(self, file, num): 
        # extract images from a video file
        loc = cv2.VideoCapture(file)
        for i in range(num):
            _, frame = loc.read()
                     
            # cv2.imshow('frame', frame)
        
            fn = work_dir+"frames/"
            if not os.path.exists(fn):
                os.makedirs(fn)
            fn += str(i)+".jpg"
            # save the image
            cv2.imwrite(fn, frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    
    # smooth the images,
    # perform thresholding for the bi-modal images,ane was c
    # morphological close and open
    # !!! only for segmentation of images with one modules
    def segment(self, img):
        gray = cv2.GaussianBlur(img,(3,3),0)    
        _,result = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        
        # adaptive thresholding -- not very good
        #result = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,801,2)
        kernel = np.ones((11,11),np.uint8)
        opened = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel)
    

        
        # detect connected components
        cc = cv2.connectedComponentsWithStats(opened,8,cv2.CV_32S)
        #print('num of labels: ', cc[0])
        #print('label matrix: ', cc[1])
        #print('stats matrix: ', cc[2])
        #print('the centroid matrix: ', cc[3])
        
        # find the largest area
        ind = np.argsort(cc[2][:,-1])[-2] # exclude background label
        max_component_origin = (cc[1]==ind)*img
        max_component_gray = (cc[1]==ind)*255
        self.center_of_module = np.flip(cc[3][ind,:],0)
        
        #plt.axis('off')
        #plt.figure()
        #plt.imshow(opened,cmap='gray')
        #plt.show()
        
        self.thresh = opened
        self.module_thresh = max_component_gray
        self.module_origin = max_component_origin
        
        #corner_points = cc[1][ind,:]
        #xmin, ymin = corner_points[0:2]
        #xmax, ymax = corner_points[2:4] + corner_points[0:2]
                    
        # save segmented images
        #plt.axis('off')
        #plt.figure()
        #plt.imshow(opened,cmap='gray')
        #plt.savefig(work_dir+'3_thresh.jpg', dpi=300)
        #plt.figure()
        #plt.imshow(max_component_gray, cmap='gray')
        #plt.savefig(work_dir+'3_module.jpg',dpi=300)
        #plt.figure()
        #plt.imshow(max_component_origin, cmap='gray')
        #plt.savefig(work_dir+'3_module_original.jpg',dpi=300)        
        #cv2.imwrite(work_dir+'3_module.jpg', max_component_gray)    
        #cv2.imwrite(work_dir+'3_module_original.jpg', max_component_origin) 
        
        # return max_component_gray
        return 
    
    
    # find the sharpest image out of the image sequence
    # base on SNR
    def find_best_snr(self, img_names):
        snr = []
        for name in img_names:
            img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
            snr.append(signaltonoise(img,axis=None))
        for i, s in enumerate(snr):
            print(i,s)
    
    
    # find the sharpest image out of the image sequence
    # base on Frequency Domain Image Blur Measure
    def find_best_FM(self, img_names):
        fm = []
        for name in img_names:
            img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
            m,n = img.shape
            img = np.float32(img)
            F = cv2.dft(img,flags=cv2.DFT_COMPLEX_OUTPUT)
            Fc = np.fft.fftshift(F) # shifting the origin of F to center
            AF = np.abs(Fc)
            M = AF.max()
            thres = M/1000
            Th = (F>thres).sum()        
            FM = np.float(Th)/(m*n)
            
            fm.append(FM)
            
        return fm

    
    def find_best_img(self, img_nums):   
        files = []
        for i in range(img_nums):
            name = img_dir+str(i)+'.jpg'
            files.append(name)
        FMs = self.find_best_FM(files)
        best_num= np.argmax(FMs)
        best_img = cv2.imread(files[best_num],cv2.IMREAD_GRAYSCALE)
        
        print(best_num)
        for i,img in enumerate(np.sort(FMs)):
            print(i,img)
        plt.imshow(best_img,'gray')
        plt.tight_layout()
        plt.show()
        
    
    def canny_detection(self, file):  
        img = cv2.imread(file)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        canny = cv2.Canny(gray, 50, 100, apertureSize=3)
        #edges = cv2.Canny(gray,  50, 100)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        cnt = contours[0]
        for c in  contours:
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        a = []

        for c in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if (len(approx) == 4):
                cv2.drawContours(img,[c],0,(0,255,0),2)
            break
                
                    #cv2.putText(img, 'o', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

        #cv2.drawContours(img, contours, -1, (0,255,0),2)
        
        cv2.imshow('test', img)
        cv2.waitKey(0)
        
        
    def houghLineP(self, img):
        edges = cv2.Canny(img, 50, 100, apertureSize=3)   
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,30,10)
        origin = cv2.imread(work_dir+'3.jpg')
        for l in lines:
            x1,y1,x2,y2 = l.tolist()[0]
            cv2.line(origin,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", origin)
            cv2.waitKey(900)
            
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", origin)

    # detect the lines and compute the four corner points
    def houghLine(self, img):
        # img is binary, so we need to save it first and open it in grayscale
        temp_loc = work_dir+'temp.jpg'
        cv2.imwrite(temp_loc, img)
        img = cv2.imread(temp_loc, 0)
        
        edges = cv2.Canny(img, 50, 100, apertureSize=3)
        # !!!the final parameter can influence how many lines will be detected 
        #lines = cv2.HoughLines(edges,1,np.pi/180,50) 
        lines = cv2.HoughLines(edges,1,np.pi/180,30)
        corners = self.findCorners(lines)
        origin = cv2.cvtColor(self.origin_img, cv2.COLOR_GRAY2RGB)
        for line in lines:
            rho,theta = line.tolist()[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            
            # to plot the hough lines
            cv2.line(origin,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", origin)
            cv2.waitKey(500)    
             
        #cv2.imshow('hough', img)
        #cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", origin)
        #cv2.imwrite(work_dir+'hough.jpg', edges)
        #cv2.waitKey(0)
        
        return corners


    # input parameter is the array of lines returned by Hough transform
    # assume that we hae all four lines
    def findCorners(self, lines):
        # convert into a simple ndarray
        rho = lines[:,:,0].flatten() 
        theta = (180/np.pi)*lines[:,:,1].flatten()
        # situation: process negative rho, and theta around 90 and theta around 180
        rho_t = np.abs(rho)
        theta_t = np.abs(theta-90)
        
        lines_org = np.stack((rho,theta), axis=1)     
        lines_new = np.stack((rho_t,theta_t), axis=1)
        
        # differentiate these lines
        #rho_0 = (rho.max()-rho.min()) / 2.0  # wrong! should no do this
        theta_0 = (lines_new[:,1].max()+lines_new[:,1].min()) / 2.0 
        ind_hor = lines_new[:,1] < theta_0
        ind_ver = ~ind_hor
        
        #ind = np.argsort(lines[:,1])
        #lines_new = lines[ind,:] 
        
        hor_lines = lines_new[ind_hor,:]
        hor_lines_org = lines_org[ind_hor,:]
        ver_lines = lines_new[ind_ver,:]
        ver_lines_org = lines_org[ind_ver,:]
        
        rho_0 = (hor_lines[:,0].max()+hor_lines[:,0].min()) / 2.0
        ind_up = hor_lines[:,0] < rho_0
        # we can use median to compute the upper and lower lines
        # TODO: better idea is to somehow average them
        i1 = np.argsort(hor_lines_org[ind_up,0])[median_low(range(sum(ind_up)))]
        i2 = np.argsort(hor_lines_org[~ind_up,0])[median_low(range(sum(~ind_up)))]
        upper = hor_lines_org[ind_up,:][i1,:]
        down = hor_lines_org[~ind_up,:][i2,:]
        
        rho_1 = (ver_lines[:,0].max()+ver_lines[:,0].min()) / 2.0
        ind_left = ver_lines[:,0] < rho_1
        # use median or 
        # TODO: somehow use mean
        i3 = np.argsort(ver_lines_org[ind_left,0])[median_low(range(sum(ind_left)))]
        i4 = np.argsort(ver_lines_org[~ind_left,0])[median_low(range(sum(~ind_left)))]
        left = ver_lines_org[ind_left,:][i3,:]
        right = ver_lines_org[~ind_left,:][i4,:]
        
    
        # corectly detected the right lines
        #upper=right.copy()
        #plotLines(upper[0],upper[1])
        
        corners = np.zeros((4,2))
        line_params = np.stack((left,upper,right,down,left), axis=0)#.astype(int)
        
        # in the order: upper-left, upper-right, down-right, down-left 
        for i in range(4):
            r1, t1 = line_params[i,:]
            r2, t2 = line_params[i+1,:]
            
            t1 *= np.pi/180.
            t2 *= np.pi/180.
            a = np.array([[np.cos(t1),np.sin(t1)],[np.cos(t2),np.sin(t2)]])
            b = np.array([r1,r2])        
            corners[i,:] = np.linalg.solve(a,b)
            
        return corners
    
    def plotLines(self, rho, theta):
        theta *= (np.pi/180.)    
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
     
        origin = cv2.imread(work_dir+'3.jpg')
        cv2.line(origin,(x1,y1),(x2,y2),(0,0,255),1)
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", origin) 
        cv2.waitKey(0)
        return None
    
    
    # similar result as contour detection
    def find_corners(self, img):
        img = np.float32(img)
        dst = cv2.cornerHarris(img,2,3,0.04)
        dst = cv2.dilate(dst,None)
        
        img1 = cv2.imread(work_dir+'3_module_original.jpg')
        
        img1[dst>0.01*dst.max()]=[0,0,255]
        cv2.imshow('dst',img)
        cv2.waitKey(0)
    
    
    # find the contours
    # find the corner points of the module
    def find_contour(self, img):
        im, contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        for i in contours:
            if cv2.contourArea(i) > cv2.contourArea(cnt):
                cnt = i
    
        appx = cv2.approxPolyDP(cnt,0.001,True)
        #x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        img1 = cv2.imread(work_dir+'3_module.jpg')
        
        #cv2.drawContours(img1, cnt, -1, (0,0,255), 1)
        cv2.drawContours(img1, [appx], -1, (0,0,255), 1)
       
        cv2.imshow('rectangle', img1)
        cv2.waitKey(0)
     
    # perform perspective transformation   
    def perspective(self, img, corners, dims):
        # test -- works fine!
    #     points1 = np.float32([[188,107], [367,110], [144,380], [360,400]])
    #     points2 = np.fprocessingloat32([[144,380*6/10],[360,400*6/10],[144,380], [360,400]])
    #     #points2 = np.float32([[0,0],[512,0],[0,512],[512,512]])
    #     P = cv2.getPerspectiveTransform(points1,points2)
    #     dst = cv2.warpPerspective(img, P, (512,512))
    #     plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Input')
    #     plt.subplot(122),plt.imshow(dst,cmap='gray'),plt.title('Output')
    #     plt.show()
        ratio = dims[1]*1.0/dims[0] 
        height = corners[:,1].max() - corners[:,1].min()
        corners = np.float32(corners)
        corners_new = np.float32([[0,0],[height*ratio,0],[height*ratio,height],[0,height]])
        Proj = cv2.getPerspectiveTransform(corners, corners_new)
        dst = cv2.warpPerspective(img, Proj, (int(height*ratio),int(height)))
    
        self.persp_img = dst
        self.cell_img = self.persp_img.copy()
        

    #TODO: we can aslo add extra parameters to tune a little of the grid due to the
    # error of transformation    
    def splitCells(self, img, dims):
        height, width = img.shape
        h, w = dims
        s1, s2 = height*1.0/h, width*1.0/w
        x1, x2 = 0, width
        for i in range(int(h)):
            y1 = int(i*s1)+1
            y2 = y1 
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
            #cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)         
        
        y1, y2 = 0, height 
        for i in range(int(w)):
            x1 = int(i*s2)
            x2 = x1 
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
            #cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img)
        
        
        ##self.cell_img = img
        #cv2.imwrite(work_dir+'result.jpg',img)
        #cv2.waitKey(0)
        
         
    
    def save_all(self):
        dir = result_dir+self.img_name
        if not os.path.exists(dir):
            os.makedirs(dir)
            
        dir += '/'+self.img_name 
        cv2.imwrite(dir+'_1origin.jpg', self.origin_img)
        cv2.imwrite(dir+'_2thresh.jpg', self.thresh)
        cv2.imwrite(dir+'_3module_thresh.jpg', self.module_thresh)
        cv2.imwrite(dir+'_4module_origin.jpg', self.module_origin)
        cv2.imwrite(dir+'_5perspective.jpg', self.persp_img)
        cv2.imwrite(dir+'_6cells.jpg', self.cell_img)
        
    # img - grayscale image
    def test_bad_img(self, img):
        pass
    
    def process_multiple_images(self, filenames):
        for name in filenames:
            video = video_process(name)
            video.segment(video.origin_img)
    
            #splitCells(img0)
            corners = video.houghLine(video.module_thresh)
            video.perspective(video.origin_img, corners, dims)
            video.splitCells(video.cell_img, dims)
            video.save_all()


if __name__ == "__main__":  
    #video = video_process(work_dir+'160_r2.jpg')
    video = video_process(test_img)
    #video.segment(video.origin_img)
    
    #splitCells(img0)
    #corners = video.houghLine(video.module_thresh)
    #video.perspective(video.origin_img, corners, dims)
    #video.splitCells(video.cell_img, dims)
    #video.save_all()

    #video.splitCells(video.persp_img, dims)
    
    #img1 = cv2.imread(work_dir+'2.jpg', 0)
    #plt.hist(img1.ravel(),256,[0,256]); plt.show()
    #img2 = cv2.imread(work_dir + '3.jpg', 0)
    #corr = signal.correlate2d(img1, img2, boundary='symm', mode='same')
    #feature.register_translation(img1,img2)
    #y, x = np.unravel_index(np.argmax(corr), corr.shape)
    #print(x,y)
    #plt.imshow(corr,cmap='gray')
    
    
    
    video.canny_detection(work_dir+'raw_img_1.jpg')
    plt.show()
  
    print("finish")


