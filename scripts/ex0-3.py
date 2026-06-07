import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

def imjshow(img,format=".jpg",**kwargs):
    img=cv2.imencode(format,img)[1]
    img=display.Image(img,**kwarges)
    dispolay.Image(img)

def draw_contours(img, contours ,ax):
    ax.imshow(img)
    ax.set_axis_off()
    
    for i, cnt in enumerate(contours):
        cnt=cnt.squeeze(axis=1)
        
        ax.add_patch(plt.Polygon(cnt , color="b",fill=None,lw=2))
        
        ax.plot(cnt[:, 0],cnt[:, 1],"ro",mew=0, ms=4)
        
        ax.plot(cnt[0][0], cnt[0][1],"ro",mew=0,ms=4)
        
        ax.text(cnt[0][0],cnt[0][1], i,color="r",size="20", bbox=dict(fc="w"))
        
        area=cv2.contourArea(cnt)
        print(f"contour: {i}, area: {area}")

#convert "kadai_02.JPG" -strip "kadai_02.JPG"
img=cv2.imread("kadai_01.JPG")
img=cv2.resize(img,(800,600))
#cv2.imshow("test",img)
imgg=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

blur=cv2.blur(imgg,(3,3))
#cv2.imshow("test1",blur)

ret, th=cv2.threshold(blur,33,255,cv2.THRESH_BINARY)
#cv2.imshow("test2",th)

contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours=list(filter(lambda x: cv2.contourArea(x)>10,contours))
#contours=list(filter(lambda x: cv2.contourArea(x)> 200,contours))
fig, ax=plt.subplots(figsize=(8,8))
draw_contours(th,contours,ax)



plt.show()
cv2.waitKey(0)