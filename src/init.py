
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage import exposure
import scipy.misc

import os
from os.path import dirname
from os.path import realpath



# configuration of all parameters

# the same directory as the zae_test
#working  = dirname(dirname(dirname(realpath(__file__)))) + '/'
working = '/home/jingpeng/work_dir/'

# first-level directory
dirs = dict(
    work_dir = working,
    origin_dir = working+'origin/',
    module_dir = working+'module/',
    persp_dir = working+'persp/'
    )

params = dict(
    dims = [6,10], # height,width
    rows = 512,
    cols = 640,
    offset = 0,
    img_num = 0
    )

# create the folders if necessary
for d in dirs:
    if not os.path.exists(dirs[d]):
        os.makedirs(dirs[d])