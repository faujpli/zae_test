'''
Created on Apr 13, 2018

@author: jingpeng
'''


from setup import *

fn0 = match_modules+'1_1.jpg'
fn1 = match_modules+'3_1.jpg'
fn2 = match_modules+'20_1.jpg'
fn3 = match_modules+'12_1.jpg'
fn3 = match_modules+'639_2.jpg'

names = []
for i in range(100,110):
    names.append(match_modules+str(i)+'_1.jpg')


def ecc_motion(fn1, fn2):
    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0) 
    
    size= img1.shape
    mat = np.eye(2,3,dtype=np.float32)
    mode = cv2.MOTION_EUCLIDEAN
    iter = 500
    eps = 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iter, eps)
    
    (cc,mat) = cv2.findTransformECC(img1,img2,mat,mode,criteria)
    print(cc)
    #print(mat)

def rigid_motion(fn1, fn2):
    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0) 
    
    #retval = cv2.estimateRigidTransform(img1,img2,False)
    shape = (1, 10, 2)
    source = np.random.randint(0, 100, shape).astype(np.int)
    target = source + np.array([1, 0]).astype(np.int)
    transformation = cv2.estimateRigidTransform(source, target, False)
    
    print(transformation)

if __name__ == "__main__":
    #motion(fn0, fn3)
    for name in names:
        ecc_motion(fn0, name)

        
        
        
        
        
        
        
        