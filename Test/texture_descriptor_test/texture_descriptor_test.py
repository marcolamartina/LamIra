import cv2
import matplotlib.pyplot as plt
import os

def get_gradient(src):

    sobelx = cv2.Sobel(src,cv2.CV_32F,1,0,ksize=5)
    sobely = cv2.Sobel(src,cv2.CV_32F,0,1,ksize=5)

    grad = sobelx + sobely
    mag = cv2.magnitude(sobelx, sobely)  # so my Mat element values could be anything between 0 and 1???
    ori = cv2.phase(sobelx, sobely, True) # so my Mat element values could be anything between 0 and 360 degrees???
    return [grad, mag, ori]

path=os.path.join(os.path.dirname(__file__),"palla-da-tennis.jpg")
src=cv2.imread(path)
#cv2.imshow("img",src)
#cv2.waitKey(0)

grad_res = get_gradient(src)

# number of bins is 100 from 0 to 1. Ie, 0.001, 0.002, ... 1.000 
# am I correct?
mag_hist = cv2.calcHist([grad_res[1]],[0],None,[1],[0,100]) 

ori_hist = cv2.calcHist([grad_res[2]],[0],None,[360],[0,360]) 

plt.plot(mag_hist)
plt.xlim([0,1])

#plt.plot(ori_hist)
#plt.xlim([0,360])

plt.show()

