#cd C:\Users\toneg\coin\opencv
#python nine_2.py
import cv2
import math
import numpy as np
from PIL import Image
import ex1
import eight
import tensorflow as tf

def main():
    for num in range(10):
        filename, kadainum = kadai_select(num+1)
        img_src, img_bin = otsu(filename)#戻値:大津の二値化した画像
        img_src = kukei(img_src, img_bin, kadainum)

    #eight.imgshow(img_bin,"otsu")
    #eight.imgshow(img_src, "kukei")
    cv2.waitKey()#delay[msec]
    cv2.destroyAllWindows()

def kadai_select(num):
    while True: 
        #filenum = int(input("課題番号を入力して下さい(1-10):"))
        filenum =   num
        if 0 < filenum and filenum < 10:
            print(f'課題番号={filenum}')
            filename = f'kadai_0{filenum}.JPG'
            return filename, filenum
        if filenum == 10:
            filename = 'kadai_10.JPG'
            return filename, filenum
        
def otsu(filename):
    img_src = cv2.imread(filename)#kadai01.JPG=4608:3456=4:3
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_RGB2GRAY)#グレースケール化
    ret,img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)#大津の二値化ret:閾値
    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)#8近傍
    img_close5 = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, element8, iterations = 5)#クロージング
    img_bin = cv2.morphologyEx(img_close5, cv2.MORPH_OPEN, element8, iterations = 5)#オープニング
    return img_src, img_bin

def kukei(img_src, img_bin, kadainum):
    height, width, ch = img_src.shape#rows:縦画素数, cols:横画素数, ch:チャンネル数
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )#輪郭検出contours:輪郭
    for num in range(len(contours)):
        if len(contours[num]) > 200:
            cv2.drawContours(img_src, contours, num, (0, 0, 255), 2)#B:0, G:0, R:255輪郭描画 
            con = contours[num]#num番の輪郭について
            print(f'\n輪郭の番号:{num}')
            print(f'(x,y)が{len(con)}コ')
            con = contours[num]#num番の輪郭について
            x_max = 0
            x_min = height
            y_max = 0
            y_min = width
            for i in range(len(con)):
                x=con[i][0][0]
                y=con[i][0][1]
                if x_max < x:
                    x_max=x
                elif x_min > x:
                    x_min = x
                if y_max < y:
                    y_max = y
                elif y_min > y:
                    y_min = y
            cv2.rectangle(img_src,(x_max, y_max),(x_min, y_min),(0,255,0),thickness = 5)#外接矩形
            tate = x_max - x_min
            yoko = y_max - y_min
            print(f'たて:{tate}, よこ:{yoko}')
            area = tate*yoko
            print(f'area={area}')
            text=f'No.{num}'
            #cv2.putText(img_src, text, (x_min,y_min-7),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), thickness = 2)
            #ex1.save(img_src,f'kadai_{kadainum}/kukei_kadai{num}')
            img_kukei = img_src[y_min : y_max, x_min : x_max]
            #ex1.save(img_kukei,f'kadai_{kadainum}/_one_kukei{num}')
            img_kukei = cv2.resize(img_kukei,(300,300))
            x = suiron2(img_kukei)
            cv2.putText(img_src, f'{x}yen', (x_min,y_min-9),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), thickness = 3)

            #ex1.save(img_kukei,f'deep/data/{kadainum}_{num}')#deeplearning用
    ex1.save(img_src,f'1225_coin/5_kadai_{kadainum}')
    return img_src

def suiron(img):
    img = img.reshape(1, 300, 300, 3)
    img = np.float32(img)/255.
    model = tf.keras.models.load_model('1225_coin/2takusan_epoch_300.h5')
    pred = model.predict(img)
    if pred < 2.5:
        pred = 1
    elif 2.5 <= pred and pred < 7.5:
        pred = 5
    elif 7.5 <= pred and pred < 30:
        pred = 10
    elif 30 <= pred and pred < 75:
        pred = 50
    elif 75 <= pred and pred < 300:
        pred = 100
    elif 300 <= pred:
        pred = 500
    return pred

def suiron2(img):
    img = img.reshape(1, 300, 300, 3)
    img = np.float32(img)/255.
    model = tf.keras.models.load_model('1225_coin/100_200_300_epoch_100.h5')
    pred = model.predict(img)
    if pred < 150:
        pred = 1
    elif 150 <= pred and pred < 250:
        pred = 5
    elif 250 <= pred and pred < 350:
        pred = 10
    elif 350 <= pred and pred < 450:
        pred = 50
    elif 450 <= pred and pred < 550:
        pred = 100
    elif 550 <= pred:
        pred = 500
    return pred

if __name__=='__main__':
    main()