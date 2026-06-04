#cd C:\Users\toneg\coin\opencv
#python ex1.py
import cv2
import math
import numpy as np
def main():
    file_name = "kadai_01.JPG"
    img_src = cv2.imread(file_name)#kadai01.JPG=4608:3456=4:3
    #img_src = resise1(img_src)#リサイズ
    #img_dst = cv2.flip(img_src, flipCode = 0)#垂直反転
    img_dst = cv2.cvtColor(img_src,cv2.COLOR_RGB2GRAY)#グレースケール化
    img_bgr = cv2.split(img_src)#複数色チャンネルの分割
    #B->R, G->B, R->Gに変更
    img_rbg = cv2.merge((img_bgr[2],img_bgr[0],img_bgr[1]))
    imgshow(img_rbg,"rgb")
    save(img_rbg,'kadai_1\irekae')#画像を保存.画像変数,ファイルパス
    img_hsv = cv2.cvtColor(img_src, cv2.COLOR_RGB2HSV)
    img_hsv2 = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    imgshow(img_hsv,"hsv")
    imgshow(img_hsv2,"hsv2")
    save(img_hsv,'kadai_1\hsv')
    print("(高さ, 幅, 色)="+str(img_hsv.shape))

    img_gray = cv2.cvtColor(img_src,cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(3,3),0)
    ret,img_bw = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)#大津の二値化
    imgshow(img_bw,"niti")
    print(f'ret={ret}')
    #maxdevisor(4608,3456)
    #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)#リサイズ可能
    #img_src = resise1(img_src)
    #img_dst = resise1(img_dst)
    #cv2.imshow("src", img_src)#winname,mat
    #cv2.imshow("dst", img_dst)
    imgshow(img_src,"src")
    imgshow(img_dst,"dst")
    save(img_dst,'kadai_1\gray')#画像を保存.画像変数,ファイルパス
    #cv2.imwrite('kadai_1\gray.jpg', img_dst)#保存
    cv2.waitKey()#delay[msec]

def save(img,filename):
    cv2.imwrite(f'{filename}.jpg', img)#保存

def resise1(img):
    i = 2
    x=400*i
    y=300*i
    img=cv2.resize(img,(x,y))
    return img

def maxdevisor(a,b):
    divisor = math.gcd(a,b)
    print(f'{a}と{b}の最大公約数 = {divisor}')

def imgshow(img,a):
    img = resise1(img)
    cv2.imshow(f'{a}',img)#winname,mat
    

if __name__=='__main__':
    main()