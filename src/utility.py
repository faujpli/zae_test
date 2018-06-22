import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
import os
from init import params, dirs



class Util:
    Prob = 0.8

    # read raw video data
    # input: raw video name with full path
    @staticmethod
    def read_raw_video(raw_name, dir_to_save):
        rows = params['rows']
        cols = params['cols']
        raw_size = os.path.getsize(raw_name)
        img_num = raw_size / (rows*cols*2) # 4 (32 bits) or 2 (16 bits)
        num = int(img_num)
        
        raw = open(raw_name, 'rb')
        f = np.fromfile(raw, dtype=np.uint16, count=rows*cols*num) # rows*cols*(offset+num)
        
        fm = []
        factor = 1    
        for i in range(0,num):
            if i==200:
                break
            start = rows*cols*i
            end = rows*cols*(i+1)
            img = f[start:end].reshape(rows,cols)

            # contrast stretching
            p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))
                        
            fm1 =format(100*Util.compute_quality(img), '.5f')
            fm.append(str(i+1)+'\t'+fm1) # start with index 1
            
            scipy.misc.imsave(dir_to_save+str(i+1)+'.jpg', img) # save to jpg file  
                         
        # save the quality factors of the images to a file
        with open(dirs['work_dir']+os.path.basename(raw_name).split('.')[0]+'_quality.txt', 'w') as f:
            for s in fm:
                print(s, file=f)
    
    
    # Compute the probability of the two images being same module
    @staticmethod
    def template_matching(template, img):
        h,w  = template.shape
        res= cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        
        # matching for only one object
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      
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


    # given the classfication results, compute the best quality one
    def select_best(classes, dir_to_comp):
        best_img_num = []
        resToFile = []
        for c in classes:
            img_names = classes[c]
              
            files = []
            for n in img_names:
                name = raw_img_module + n + '.jpg'
                files.append(name)
            FMs = find_best_FM(files)
            num = np.argmax(FMs)
            best_img_num.append(img_names[num])
            
            #with open(work_dir+'scores.txt', 'a') as f:  
            #    for i in reversed(np.argsort(FMs)[-10:]):
            #        print(img_names[i], FMs[i], file=f)
            #    #print('\n---------------------\n', file=f)
            
            s = 'The best quality image for module '+str(c)+' is '+str(img_names[num])+'.jpg'
            resToFile.append(s)
            print(s)
        
            filename = img_dir+img_names[num].split('_')[0]+'.jpg'
            
            #img = cv2.imread(filename,0)
            #cv2.imshow(s, img)
            #cv2match_modules.waitKey(500)
        
        
        # save the resulting best image index information    
        with open(match_res+'best_image.txt', 'w') as f:
            print(best_img_num, file=f)
            for i in resToFile:
                print(i+'\n', file=f) 
        
        return best_img_num



    
    