#cd C:\Users\toneg\coin\opencv
#python video.py
import cv2
import math
import numpy as np

def main():
    #cv2.namedWindow('src')
    cv2.namedWindow('dst')
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)
    while True:
        ret, img_src = cap.read()#カメラ画像読み込みret:TorF, img_src:image
        img_dst = cv2.flip(img_src, 1)#左右反転
        #cv2.imshow('src', img_src)
        cv2.imshow('dst', img_dst)
        ch = cv2.waitKey(1)
        if ch == ord('q'):                                                                                                                                                                                  
            break
    cv2.destroyAllWindows()    

if __name__=='__main__':
    main()