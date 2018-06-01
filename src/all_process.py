
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc
from setup import *
from matching import segment_modules



# read raw video data
# input: raw video name with full path
def read_raw_video(raw_name):
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
        
        fm1 =format(1000*compute_quality(img), '.5f')
        fm.append(str(i)+' '+fm1)
        
        segment(img, )
        
        #scipy.misc.imsave(raw_dir+str(i)+'.jpg', img) # save to jpg file  
        
        # show the quality factor on the images
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, text, (0, 25),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        #cv2.imshow('Image', img)
        #cv2.waitKey(100)
        
    # save the quality factors of the images to a file
    with open(work_dir+os.path.basename(raw_name).split('.')[0]+'_quality.txt', 'w') as f:
        for s in fm:
            print(s, file=f)


# segment all single modules 
def segment_modules(img_dir, image_indices):
    for i in image_indices:
        img = cv2.imread(img_dir+str(i)+'.jpg', 0)
        save_path = raw_img_module+str(i)+'_'

def segment(img): #,save_path):
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


# tracking all 
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


# transform all the single module images 
# TODO: add the case for only illustration - no need to perform transformation
def perspective_all(dir_module, only_for_center):
    # for all the modules iamges, compute the perspective transform
    for filename in sorted(os.listdir(dir_module), key=lambda x: (int(x.split('_')[0]), int(x.split('.')[0].split('_')[1]))):
        video = video_process(dir_module+filename)
        video.segment(video.origin_img)
        
        corners = video.houghLine(video.module_thresh)
        video.perspective(video.origin_img, corners, dims)
        #cv2.imwrite(match_persp+os.path.splitext(filename)[0] + '.jpg', video.persp_img)
        
        # save the centroids of these modules for later use, i.e. 
        # illustration of modules @show_modules
        # and also acting as a reference point for transformation
        center = video.center_of_module
        n1, _ = filename.split('_')

        if n1 in centers:
            centers[n1].append(center)
        else:
            centers[n1] = [center]
        
        
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

# find out the images that correspond to the same module with very high probability, e.g. 0.8
def classfy_modules(save_with_prob):
    # performing classification for the modules
    dir_to_match = match_persp_full
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

            res = template_matching(template, img)
            #print(res)
            if res > best_match:
                best_match = res
                best_key = key
        
        if best_match > 0.8:
            if save_with_prob == True:
                val = '('+val+', '+str(best_match)+')' # with probabilities
            results[best_key].append(val)
        else:      
            if save_with_prob == True:
                val = '('+val+', '+str(best_match)+')' # with probabilities           
            results[len(results)+1] = [val]                      
           
    return results     
