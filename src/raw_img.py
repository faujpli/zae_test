'''
Created on Mar 9, 2018

@author: jingpeng
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
from setup import *
from matching import segment_modules


# compute the image quality based on its sharpness
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
    #print(FM)
            
    return FM


# compute the amoutn of blur in the image
def compute_blur(img):
    fm = cv2.Laplacian(img, cv2.CV_64F).var()
    #print(fm)
    
    # show the image
    #cv2.putText(img, str(fm), (100, 300),
    #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
    #cv2.imshow("Image", img)
    #cv2.waitKey(300)
    return fm
        
                
# read the raw image data and manipulate the memory
def test_raw_img():
    raw = open(work_dir+'test.raw', 'rb')
    f = np.fromfile(raw, dtype=np.uint32, count=rows*cols*3)  
    im = f.reshape(rows,cols,3) #notice row, column format
        
    p2, p98 = np.percentile(im, (2,98))
    im = exposure.rescale_intensity(im, in_range=(p2, p98))   
    
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(im[:,:,1],cmap='gray')
    
    plt.show()


# read in raw video data
def test_raw_video(raw_name):
    # number of images
    raw_size = os.path.getsize(raw_name)
    img_num = raw_size / (rows*cols*2) # 4 (32 bits) or 2 (16 bits)
    num = int(img_num)
    
    raw = open(raw_name, 'rb')
    f = np.fromfile(raw, dtype=np.uint16, count=rows*cols*(offset+num))
    # normalize the intensities to be in [0,255]
    f = 255.*(f - f.min())/(f.max()-f.min())
    fm = []    
    for i in range(offset,offset+num):
        start = rows*cols*i
        end = rows*cols*(i+1)
        img = f[start:end].reshape(rows,cols)
        #qf = format(compute_quality(img), '.6f')
        #qf =format(compute_blur(img), '.4f')
        
        # contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))
        
        fm1 =format(1000*compute_quality(img), '.5f')
        fm2 =format(compute_blur(img), '.5f')
        
        text = fm1+' '+fm2
        fm.append(str(i)+' '+text)
        

        
        #scipy.misc.imsave(raw_dir+str(i)+'.jpg', img) # save to jpg file  
        
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, text, (0, 25),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        #cv2.imshow('Image', img)

        #cv2.waitKey(100)
    
    with open(work_dir+os.path.basename(raw_name).split('.')[0]+'_quality.txt', 'w') as f:
        for s in fm:
            print(s, file=f)
            
            
            
# save raw video to jpg images
def save_raw_to_jpg(raw_name):
    raw_size = os.path.getsize(raw_name)
    img_num = raw_size / (rows*cols*2) # 4 (32 bits) or 2 (16 bits)
    num = 100
    offset = 1000
    
    raw = open(raw_name, 'rb')
    f = np.fromfile(raw, dtype=np.int16, count=rows*cols*(offset+num))
    # normalize the intensities to be in [0,255]
    f = 255.*(f - f.min())/(f.max()-f.min())
    for i in range(offset,offset+num):
        start = rows*cols*i
        end = rows*cols*(i+1)
        img = f[start:end].reshape(rows,cols)
        
        # contrast stretching
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))     
        
        scipy.misc.imsave(raw_img_dir+str(i)+'.jpg', img) # save to jpg file




# read raw image data and convert it to jpg image  
# Given: file path, dimensions
# Return: the covnerted image
def raw_to_raw(raw_name, rows, cols):
    raw = open(raw_name, 'rb')
    f = np.fromfile(raw, dtype=np.uint16)
    img = 255.*(f - f.min())/(f.max()-f.min())
    img = np.reshape(img, (rows,cols)).astype(np.uint16)
    
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    dir = os.path.dirname(raw_name)+'/'
    name = os.path.basename(raw_name).split('.')[0]
    scipy.misc.imsave(dir+name+'.jpg', img)
    
    indices = []
    indices.append(name)
    segment_modules(dir, indices)
    
    #cv2.imshow('img',img)
    cv2.waitKey(0)
    
    # save as a jpg image
    jpg_name = raw_name.split('.')[0]
    jpg_name += '.jpg'
    #scipy.misc.imsave(jpg_name, img)
    print("good")
    return img_dir+'test.raw'

# short program to test how image size influence the image quality 
def test_qf(filename):
    img = cv2.imread(filename, 0)
    l = min(img.shape)
    for i in range(1, l, 15):
        roi = img[:i,:]
        qf = format(compute_blur(roi), '0.4f')
        print(qf)

        cv2.imshow('img', roi) 
        cv2.waitKey(200)

# short program to test the effectiveness of the function - compute_blur
def test_compute_blur():    
    filenames = sorted(os.listdir(match_modules), key=lambda x: (int(x.split('_')[0]), int(x.split('.')[0].split('_')[1])))
    fm = []
    for name in filenames:
        img = cv2.imread(match_modules+name,0)
        fm1 =format(10000*compute_quality(img), '.7f')
        fm2 =format(compute_blur(img), '.3f')
        
        text = fm1+' '+fm2
        fm.append(name+' '+text)
        
        print(name,text)
        
        
        # show the image
        cv2.putText(img, text, (0, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(150) & 0xFF == ord('q'):
            break
        
    with open(work_dir+'compare_image_quality.txt', 'w') as f:
        for s in fm:
            print(s, file=f)

# TODO:
# transform the raw image
# input: raw image
def raw_to_persp(img1):
    img = cv2.imread(match_res+'module 0.jpg', 0)
    
    pass
        

test_dir = '/media/jingpeng/Maxtor/raw_videos/Arzberg/'

#res = find_best_FM(im)
#test_raw_img()
#test_raw_video(work_dir+'LockIn_Velox_10m_2ms_100FPS.raw')
#raw_to_jpg(work_dir+'32bit.raw', 512, 640)
#test_qf(work_dir+'lena.png')
#test_compute_blur()
save_raw_to_jpg(test_dir+'BYD_SOlow_5ms_6,4A_Night.raw')
#raw_to_raw(work_dir+'raw_img.raw', 512, 640)


#test_vidoes_path = '/media/jingpeng/Maxtor/raw_videos/Speed/test/'
#for video_file in os.listdir(test_vidoes_path):
#    print(video_file)
#    test_raw_video(test_vidoes_path+video_file)





