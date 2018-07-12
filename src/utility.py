import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
import os
from PIL import Image
from init import params, dirs



class Util:
    Prob = 0.88
    Rawname = 'Test video:'

    # read raw video data
    # input: raw video name with full path
    @staticmethod
    def read_raw_video(raw_name, dir_to_save):
        rows = params['rows']
        cols = params['cols']
        raw_size = os.path.getsize(raw_name)
        img_num = raw_size / (rows*cols*2) # 4 (32 bits) or 2 (16 bits)
        num = int(img_num)
        Util.Rawname = os.path.basename(raw_name).split('.')[0]
        
        
        raw = open(raw_name, 'rb')
        f = np.fromfile(raw, dtype=np.uint16, count=rows*cols*num) # rows*cols*(offset+num),  np.uint32 (32bits)
        
        fm = []
        factor = 1    
        for i in range(0,num):
            start = rows*cols*i
            end = rows*cols*(i+1)
            img = f[start:end].reshape(rows,cols)

            # contrast stretching
            p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))
                        
            fm1 =format(100*Util.compute_quality(img), '.5f')
            fm.append(str(i+1)+'\t'+fm1) # start with index 1
            
            scipy.misc.imsave(dir_to_save+str(i+1)+'.jpg', img) # save to jpg file  
        
        print("origin_finished!")
                        
        # save the quality factors of the images to a file
        with open(dirs['work_dir']+Util.Rawname+'_quality.txt', 'w') as f:
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
    def classify_modules(dir_to_match, save_with_prob):
        # performing classification for the modules
        img_names = sorted(os.listdir(dir_to_match), key=lambda x: int(x.split('_')[0]))
        ref_img = img_names[0]
        ref_name = os.path.splitext(ref_img)[0] # with extension - '.jpg'
        results = {1:[ref_name]} # default: for template
        if save_with_prob == True: # for later consistency
            ref_name = '('+ref_name+', 1.0)'           
            results_save = {1:[ref_name]}
        template = cv2.imread(dir_to_match+ref_img,0)
        
        for filename in img_names:
            img = cv2.imread(dir_to_match+filename,0)
            #img = cv2.equalizeHist(img)
            val = os.path.splitext(filename)[0]
            best_match = 0.
            best_key = 0
            for key in results:
                #template = cv2.imread(dir_to_match+results[key][0]+'.jpg', 0) # not so good
                #if save_with_prob == True:
                    #temp_name = results_save[key][-1].split(',')[0].split('(')[1] # if with prob
                temp_name = results[key][-1] # default: without prob 
                template = cv2.imread(dir_to_match+temp_name+'.jpg', 0) # much better
                
                res = Util.template_matching(template, img)
                #print(res)
                if res > best_match:
                    best_match = res
                    best_key = key
            
            if best_match > Util.Prob: # control how similar the module are
                results[best_key].append(val)
                if save_with_prob == True:
                    val = '('+val+', '+str(best_match)+')' # with probabilities
                    results_save[best_key].append(val)                
            else:
                results[len(results)+1] = [val]     
                if save_with_prob == True:
                    val = '('+val+', '+str(best_match)+')' # with probabilities
                    results_save[len(results_save)+1] = [val]
                          
        
        # save the results
        Util.write_txt(results_save,save_with_prob)
        best_images = Util.select_best(results)
        Util.save_best_img(best_images)
    
        print('classify_finished')
               
        return results     
    
    #save the classification result to a text file
    @staticmethod
    def write_txt(results,save_with_prob):
        filename = 'matching_results'
        if save_with_prob == True:
            filename += '_with_prob'
        filename += '.txt'
        with open(dirs['work_dir']+filename, 'w') as f:
            for i in results:
                #print(os.path.splitext(i[0])[0]+': ', i[1], file=f)
                s = results[i][0]
                for n in results[i][1:]:
                    s += ', '+n
                print(str(i)+': '+s, file=f)



    # given the classification results, compute the best quality one
    @staticmethod
    def select_best(classes):
        best_images = []
        resToFile = []
        for c in classes:
            img_names = classes[c]
              
            files = []
            for n in img_names:
                name = dirs['persp_dir'] + n + '.jpg'
                files.append(name)
            FMs = Util.compute_FMs(files)
            num = np.argmax(FMs)
            best_images.append(str(img_names[num]))
            
            
            s = 'The best quality image for module '+str(c)+' is '+str(img_names[num])+'.jpg'
            resToFile.append(s)            
            print(s)

        # save the resulting best image index information    
        with open(dirs['work_dir']+'best_images.txt', 'w') as f:
            print(Util.Rawname+':\n', file=f)
            for i in resToFile:
                print(i+'\n', file=f) 
        
        return best_images


    # compute the quality factors for a sequence
    @staticmethod
    def compute_FMs(file_names):
        FMs = []
        for name in file_names:
            img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
            FM = Util.compute_quality(img)            
            FMs.append(FM)
            
        return FMs
    
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
    
    @staticmethod
    def save_best_img(best_images):
        for name in best_images:
            img = cv2.imread(dirs['persp_dir']+name+'.jpg',0)
            cv2.imwrite(dirs['work_dir']+name+'.jpg', img)
            #tiff.save(dirs['work_dir']+name, img)
            
    @staticmethod
    def save_tiff(raw_name, dir_to_save):
        rows = params['rows']
        cols = params['cols']
        raw_size = os.path.getsize(raw_name)
        img_num = raw_size / (rows*cols*2) # 4 (32 bits) or 2 (16 bits)
        num = int(img_num)
        Util.Rawname = os.path.basename(raw_name).split('.')[0]        
        
        raw = open(raw_name, 'rb')
        f = np.fromfile(raw, dtype=np.uint16, count=rows*cols*num) # rows*cols*(offset+num)
          
        for i in range(1000,num):
            if i > 1006 :
                break
            start = rows*cols*i
            end = rows*cols*(i+1)
            img = f[start:end].reshape(rows,cols)

            # contrast stretching
            p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))
            
            scipy.misc.imsave(dir_to_save+str(i+1)+'.tiff', img) # save to jpg file  
    
    