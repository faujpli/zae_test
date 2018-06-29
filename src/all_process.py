import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
import os
from utility import Util
import image_process as IP

from init import dirs


# given a directory containg all the images,
# perform segmentation
def segment_modules(img_dir, module_dir, persp_dir):
    for name in sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0])):
        i = name.split('.')[0]
        #img = cv2.imread(img_dir+name, 0)
        save_paths = [module_dir+i+'_', persp_dir+i+'_'] # .jpg will be appended in the function semgent

        ip = IP.ImageProcess(img_dir+name)
        ip.segment(save_paths)
    print('segment_finished') 


    
if __name__ == "__main__":
    #Util.read_raw_video(dirs['work_dir']+'test.raw', dirs['origin_dir'])
    #segment_modules(dirs['origin_dir'], dirs['module_dir'], dirs['persp_dir'])
    #Util.classify_modules(dirs['persp_dir'], True)    
    Util.classify_modules(dirs['work_dir']+'matching_module/', True)
    
    
    
    
    
                
    print('finished')