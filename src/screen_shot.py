'''
Created on Apr 4, 2018

@author: jingpeng
'''
import numpy as np
import cv2
import pyautogui
import imutils
import pyscreenshot as Grab
import tkinter as tk
import time

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
    print(FM)
            
    return FM


def compute_blur(img):
    fm = cv2.Laplacian(img, cv2.CV_64F).var()
        
    return fm

#img = pyautogui.screenshot()
#img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#cv2.imwrite('/home/jingpeng/1.jpg', img)

if __name__ == "__main__":

    img = Grab.grab()
    #img.save('/home/jingpeng/1.jpg')
    #img = cv2.imread('/home/jingpeng/1.jpg')
    img = np.array(img)
    r = cv2.selectROI('ROI', img, False)
    cv2.destroyWindow('ROI')
    arr = np.array(r).reshape(2,2)
    arr[1,:] += arr[0,:]
    r1,r2 = tuple(arr[0,:]),tuple(arr[1,:])
    roi_c = tuple(np.mean(arr,axis=0).astype(int))
    roi = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[r1[0]:r2[0],r1[1]:r2[1]]
    
    
    #cv2.rectangle(img,r1,r2,(1,1,1),3)
    #cv2.putText(img, '3.1415', roi_c, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,10), 1)
    
    m = tk.Tk()
    w, h = 100,50
    wid, hei = m.winfo_screenwidth(), m.winfo_screenheight()
    x,y = wid-w*2, h*2
    m.geometry('%dx%d+%d+%d' % (w,h,x,y))
    
    
    msg = tk.Message(m, text='')
    msg.config(foreground = 'red', font=('times',14),aspect=1000,justify='c')
    msg.pack()
    m.update()
    
    # fix the ROI window and continuous compute the image quality
    while True:
        m.update()
        img = Grab.grab()
        img = np.array(img)
        roi = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[r1[0]:r2[0],r1[1]:r2[1]]
        
        fm =format(compute_quality(roi), '.4f')
        print(fm)
        # display the quality values on a new window
        msg.config(text=fm)
        m.update()
        
        #time.sleep(0.5)
        if cv2.waitKey(500) == 27:
            break
        
    
    tk.mainloop()






