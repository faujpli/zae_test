import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
from setup import *


class Util:
    Prob = 0.8

    # read raw video data
    # input: raw video name with full path
    @staticmethod
    def read_raw_video(raw_name, dir_to_save):
        raw_size = os.path.getsize(raw_name)
        img_num = raw_size / (rows*cols*2) # 4 (32 bits) or 2 (16 bits)
        num = int(img_num)
        
        raw = open(raw_name, 'rb')
        f = np.fromfile(raw, dtype=np.uint16, count=rows*cols*num) # rows*cols*(offset+num)
        # normalize the intensities to be in [0,255]
        f = 255.*(f - f.min())/(f.max()-f.min())
        fm = []    
        for i in range(0,num):
            start = rows*cols*i
            end = rows*cols*(i+1)
            img = f[start:end].reshape(rows,cols)
            
            # contrast stretching
            p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))
            
            fm1 =format(1000*Util.compute_quality(img), '.5f')
            fm.append(str(i)+' '+fm1)
            
            scipy.misc.imsave(dir_to_save+str(i)+'.jpg', img) # save to jpg file  
            
                    
        # save the quality factors of the images to a file
        with open(work_dir+os.path.basename(raw_name).split('.')[0]+'_quality.txt', 'w') as f:
            for s in fm:
                print(s, file=f)
                
    
    # Compute the probability of similarity between the two images
    @staticmethod
    def template_matching(template, img):
        h,w  = template.shape
        res= cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        
        # matching for only one object
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #top_left = max_loc
        #bottom_right = (top_left[0] + w, top_left[1] + h)
        #cv2.rectangle(img,top_left, bottom_right, 255, 2)
        #print(res.max())
        #print(top_left)
        
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        #plt.show()
        
        return round(max_val,4)
    
    
    # Find out the images that correspond to the same module with very high probability, e.g. 0.85
    @staticmethod
    def classfy_modules(dir_to_match, save_with_prob):
        # performing classification for the modules
        img_names = sorted(os.listdir(dir_to_match), key=lambda x: int(x.split('_')[0]))
        ref_img = img_names[0]
        ref_name = os.path.splitext(ref_img)[0] # with extension - '.jpg'
        if save_with_prob == True: # for later consistency
            ref_name = '('+ref_name+', 1.0)' 
    
        results = {1:[ref_name]} # default: for template
        template = cv2.imread(dir_to_match+ref_img,0)
        
        for filename in img_names:
            img = cv2.imread(dir_to_match+filename,0)
            #img = cv2.equalizeHist(img)
            val = os.path.splitext(filename)[0]
            best_match = 0.
            best_key = 0
            for key in results:
                #template = cv2.imread(dir_to_match+results[key][0]+'.jpg', 0) # not so good
                if save_with_prob == True:
                    temp_name = results[key][-1].split(',')[0].split('(')[1] # if with prob
                else:
                    temp_name = results[key][-1] # default: without prob 
                template = cv2.imread(dir_to_match+temp_name+'.jpg', 0) # much better
    
                res = Util.template_matching(template, img)
                #print(res)
                if res > best_match:
                    best_match = res
                    best_key = key
            
            if best_match > Util.Prob:
                if save_with_prob == True:
                    val = '('+val+', '+str(best_match)+')' # with probabilities
                results[best_key].append(val)
            else:      
                if save_with_prob == True:
                    val = '('+val+', '+str(best_match)+')' # with probabilities           
                results[len(results)+1] = [val]                      
               
        return results     



    # compute the image quality based on its sharpness
    @staticmethod
    def compute_quality(img):
        m,n = img.shape
        img = np.float32(img)
        F = cv2.dft(img,flags=cv2.DFT_COMPLEX_OUTPUT)
        Fc = np.fft.fftshift(F) # shifting the origin of F to center
        AF = np.abs(Fc)
        M = AF.max()
        thres = M/1000
        Th = (F>thres).sum()        
        FM = np.float(Th)/(m*n)
                
        return FM

    # find the sharpest image out of the image sequence
    # base on Frequency Domain Image Blur Measure
    @staticmethod
    def find_best_fm(img_names):
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
            
            fm.appenandd(FM)
            
        return fm
    
    