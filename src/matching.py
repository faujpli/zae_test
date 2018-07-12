import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path
from video_process import *
import csv
from shutil import copyfile
from setup import *



def segment(img, save_path):
    gray =  cv2.GaussianBlur(img,(3,3),0)    
    thresh, result = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel = np.ones((11,11),np.uint8)
    open = cv2.morphologyEx(result,cv2.MORPH_OPEN,kernel)
    
    # detect connected components
    cc = cv2.connectedComponentsWithStats(open,8,cv2.CV_32S)
    #print('num of labels: ', cc[0])
    #print('label matrix: ', cc[1])
    #print('stats matrix: ', cc[2])
    #print('the centroid matrix: ', cc[3])
    #print('max components: ', np.sort(cc[2][:,-1]))
    
    stats = cc[2]
    ind = np.argsort(stats[:,-1]) # indices of all connected components
    max_components = []
    min_val = stats[ind[-2],-1]/2. # exclude background
    mod_num = 1

    for i in reversed(range(ind.size-1)):
        # check if touching the boarder: test left, right, up and down, and also min area
        if (stats[ind[i],0] != 0 and stats[ind[i],-1] >= min_val and 
            (stats[ind[i],0]+stats[ind[i],2]) != img.shape[1] and
            (stats[ind[i],1] != 0) and
            (stats[ind[i],1]+stats[ind[i],3]) != img.shape[0]):
            #max_components.append((cc[1]==ind[i])*img)
            cv2.imwrite(save_path+str(mod_num)+'.jpg', (cc[1]==ind[i])*img)
            mod_num += 1       


def template_matching(template, img):
    h,w  = template.shape
    res= cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    
    # matching for only one object
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    #print(res.max())
    #print(top_left)
    
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    #plt.show()
    return round(max_val,4)


# segment all single modules 
def segment_modules(img_dir, image_indices):
    for i in image_indices:
        #img = cv2.imread(match_dir+str(i)+'.jpg', 0)    
        #img = cv2.imread(img_dir+str(i)+'.jpg', 0)
        img = cv2.imread(img_dir+str(i)+'.jpg', 0)
        #save_path = match_modules+str(i)+'_'
        save_path = raw_img_module+str(i)+'_'

        #if not os.path.exists(save_path):
            #os.makedirs(save_path)  
        segment(img,save_path)

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
    key = None
    val = None
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


def write_txt(results,save_with_prob):
    filename = 'matching_results'
    if save_with_prob == True:
        filename += '_with_prob'
    filename += '.txt'
    with open(match_res+filename, 'w') as f:
        for i in results:
            #print(os.path.splitext(i[0])[0]+': ', i[1], file=f)
            s = results[i][0]
            for n in results[i][1:]:
                s += ', '+n
            #print(str(i)+': '+s+'\n', file=f)
            print(str(i)+': '+s, file=f)

def write_csv(results, save_with_prob):
    s = 'matching_results'
    if save_with_prob == True:
        s += '_with_prob'
    s += '.csv'
    with open(match_res+s, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for _, v in results.items():
            writer.writerow(v)

    
# load an store all the class labels information into a dictionary
# only for files without probabilities          
def read_txt(classes):
    with open(match_res+'matching_results.txt', 'r') as f:    
        for i, line in enumerate(f):
            new_line = line.replace(" ", "").rstrip()
            classes[i+1] =  new_line.split(':')[1].split(',') 

# just testing the quality of template matching
# Point1: not symetric
# Point2: should use the full transformed image, otherwise it performs poorly
# Point3: the matching probabilities decrease through time
def quality_test():          
    img1= cv2.imread(match_persp_full+'309_1.jpg',0)
    img2 = cv2.imread(match_persp_full+'184_1.jpg', 0)
    img3= cv2.imread(match_persp_full+'310_1.jpg',0)


    print(template_matching(img1, img2))
    print(template_matching(img1, img3))


def find_best_FM(img_names):
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


# input: image names
def find_best_img(img_names):   
    files = []
    for n in img_names:
        name = match_modules + n + '.jpg'
        files.append(name)
    FMs = find_best_FM(files)
    
    num= np.argmax(FMs)
    best_name = img_names[num] 
    best_img = cv2.imread(files[num],cv2.IMREAD_GRAYSCALE)
    
    #for i,img in enumerate(np.sort(FMs)):
        #print(i,img)
    
    plt.imshomatch_modulesw(best_img,'gray')
    plt.tight_layout()
    plt.show()


# add text to an image 
def add_text(): 
    img = cv2.imread(img_dir+'1.jpg',0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, '1', (200,200),font,1,(30,30,30),2,cv2.LINE_AA)
    cv2.putText(img, '2', (100,300),font,1,(30,30,30),2,cv2.LINE_AA)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    

def test_quality_dynamic():
    img_name = [match_labels+'789.jpg']
    val = find_best_FM(img_name)
    print(val)



# for illustration of class labels
def show_modules(classes, centers):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for filename in sorted(os.listdir(img_dir), key=lambda x: int(x.split('.')[0])):
        filenum = filename.split('.')[0]
        img = cv2.imread(img_dir+filename, 0)
        if filenum in centers:
            
            cents = centers[filenum]
            for i,cent in enumerate(cents):
                name = filenum+'_'+str(i+1)
                for c in classes:
                    if name in classes[c]:
                        cv2.putText(img, str(c), tuple(cent[::-1].astype(int)),font,2,(30,30,30),2,cv2.LINE_AA)
                        break
        #cv2.imshow('img', img) # showing the resulting images
        #cv2.waitKey(30)
        #cv2.imwrite(match_labels+filename,img) # saving the resulting images


# given the classfication membership, compute the best quality one
def select_best(ce for module 1 is 412_1.jpg
The best quality image for module 2 is 823_1.jpg
The best quality image for module 3 is 1575_1.jpg
The best quality image for module 4 is 2307_1.jpg
The best quality image forlasses):
    best_img_num = []
    resToFile = []
    for c in classes:
        img_names = classes[c]
        #find_best_img(img_names)
          
        files = []
        for n in img_names:
            #name = match_modules + n + '.jpg'
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


# save the resulting best images - transformed single module images
def save_best(best_img_num):
    for i,name in enumerate(best_img_num):
        src_name = match_persp_full + name + '.jpg'
        dst_name = result_dir + 'module ' + str(i) + '.jpg'
        copyfile(src_name, dst_name)        
        
        



if __name__ == "__main__":
    #test_quality_dynamic()
    #quality_test()
    #img = cv2.imread(test_img,0)
    #segment(img,'')
    
    only_for_center = True
    centers = {} # centroids for each module in a frame
    image_indices = range(0,num)#len(os.listdir(raw_img_dir)))
    segment_modules(raw_img_dir,image_indices)
    #perspective_all(match_modules, only_for_center) # have the centroids information
    perspective_all(raw_img_module, only_for_center)
    
    # save the matching results to .txt/.csv format
    save_with_prob = False
    classes = classfy_modules(save_with_prob) # True: save with probabilities
    write_txt(classes, save_with_prob)
    #write_csv(classes, save_with_prob)
    
    #classes = {} # dictionary to store the label/class information
    #read_txt(classes)
    
    
    # use either classes or results: classes == results
    #show_modules(classes, centers)
    best_img_num = select_best(classes)
    save_best(best_img_num)
   
    
    
    
    print('finished')





