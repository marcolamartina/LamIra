# importing required libraries
import numpy as np
import mahotas
from pylab import imshow, show
import os
import cv2
import eolearn.features.haralick as haralick
  
# loading image
path=os.path.dirname(__file__)
leaf=os.path.join(path,"leaf.tiff")
img = cv2.imread(leaf)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img.shape)
img[img>250]=0
#cv2.imshow("pippo",img)
#cv2.waitKey(0)

 
# getting haralick features
h_feature = haralick.HaralickTask(img)
print(h_feature)
exit(0) 
# showing the features
for i,h in enumerate(h_feature):
    print(i+1,round(h,4))

