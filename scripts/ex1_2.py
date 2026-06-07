#cd C:\Users\toneg\coin\opencv
#python ex1_2.py
import cv2
import numpy as np
import ex1
from PIL import Image
import matplotlib.pyplot as plt

def main():
    #n = kadai_num()#キーボードから入力された文字列を受け取る
    n=1
    file_name = f'kadai_0{n}.JPG'
    img_src = cv2.imread(file_name)#kadai01.JPG=4608:3456=4:3
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_RGB2GRAY)#グレースケール化
    #rat, binary_otsu = cv2.threshold(img_gray, 36, 255, cv2.THRESH_BINARY)#画素値36以上を255
    rat, binary_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)#大津の二値化
    

    #kernel = np.ones((5,5),np.uint8)

    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)#8近傍
    close = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, element8, iterations = 1)#クロージング
    open = cv2.morphologyEx(close, cv2.MORPH_OPEN, element8, iterations = 1)#オープニング

    contours, hierarchy = cv2.findContours(open, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(open, contours, -1, (0, 0, 255), 2)
    for contour in contours:
        for point in contour:
            cv2.circle(open, point[0], 3, (0, 255, 0), -1)

    cv2.imshow("Image", open)

    #binary3 = cv2.morphologyEx(binary2, cv2.MORPH_CLOSE, kernel,10)#クロージング

    #print(f'閾値は{rat}')
    #ex1.imgshow(img_gray,"gray")#画像の表示(画像ファイル, ウィンドウ名)
    #ex1.imgshow(img_src,"src")#画像の表示(画像ファイル, ウィンドウ名)
    #ex1.imgshow(binary_otsu, "binary")#表示_二値画像
    #ex1.imgshow(open, "open")#表示_オープニング後
    ex1.imgshow(open, "close")#表示_クロージング後
    ex1.save(open,f'kadai_{n}/output')#画像を保存.画像変数,ファイルパス
    #ex1.save(binary_otsu,f'kadai_{n}/binary_otsu')#画像を保存.画像変数,ファイルパス
    cv2.waitKey()#delay[msec]

def kadai_num():
    print('課題番号を入力')
    n = input()
    print(f'kadai_0{n}---')
    return n

if __name__=='__main__':
    main()