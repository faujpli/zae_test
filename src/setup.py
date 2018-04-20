'''
Created on Apr 5, 2018

@author: jingpeng
'''
import os.path
import numpy as np
import matplotlib.pyplot as plt
import cv2

# first-level directory
work_dir = '/home/jingpeng/work_dir/'
result_dir = work_dir+'results/'
img_dir = work_dir+'frames/'
raw_dir = work_dir + 'raw/'
raw_img_dir = raw_dir+'raw_img/'
match_res = work_dir+'matching_res/'

# subfolders of the folder - match_res
match_modules = match_res+'modules/'
match_persp_full = match_res+'persp_full/'
match_persp = match_res+'persp/'
match_labels = match_res+'labels/'

# test images/videos
test_img = img_dir + '179.jpg'
test_img_bin = result_dir+"120_after.jpg"
bad_img = work_dir + '1.tif'


dirs = [work_dir, result_dir, img_dir, raw_dir, raw_img_dir, match_res]
sub_dirs = [match_modules, match_persp, match_persp_full, match_labels]

# used in: video_process.py, matching.py
dims = [10, 6] # no. of cells in hight and width direction
# used in module: raw_img.py 
rows,cols = 512,640
[offset, num] = [999, 3000] # num of raw images to read

# create the folders if necessary
dirs += sub_dirs
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)




