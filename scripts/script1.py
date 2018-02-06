import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../raw_pics/1.bmp')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
invert = cv2.bitwise_not(gray)


displayed_sq = 500
fig = plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.imshow(img[:displayed_sq,:displayed_sq],cmap = 'gray')
plt.show()

print(invert.ravel())
plt.hist(invert.ravel(),256,[0,256])
plt.show()



kernel = np.ones((2,2),np.uint8)

for i in range(15, 40, 2):
    for ks in [2,3]:
        for itnum in range(1,5):
            kernel = np.ones((ks, ks), np.uint8)

            ret, thresh = cv2.threshold(gray,i,255,cv2.THRESH_BINARY)

            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = itnum)

            cv2.imwrite('../processed_pics/{}x{}-{}-{}.bmp'.format(ks,ks,itnum,i), opening)


plt.subplot(1,2,1)
plt.imshow(invert,cmap = 'gray')
plt.subplot(1,2,2)
#plt.imshow(opening,cmap = 'gray')

plt.title('Original'), plt.xticks([]), plt.yticks([])


plt.show()