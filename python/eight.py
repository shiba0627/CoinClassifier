#cd C:\Users\toneg\coin\opencv
#python eight.py
import cv2
import math
import numpy as np
from PIL import Image
import ex1

def main():
    kadai_01()
    #kadai_09()
    cv2.waitKey()#delay[msec]
    cv2.destroyAllWindows()

def kadai_09():
    kadai_09 = "kadai_09.JPG"#斜め
    img_src9 = cv2.imread(kadai_09)
    size = tuple(np.array([img_src9.shape[1], img_src9.shape[0]]))
    print(f'test1={img_src9.shape[1]},{img_src9.shape[0]},size={size}')
    #pts1 = np.float32([[160,479],[480,479],[480,240],[160,240]])
    pts1 = np.float32([[100,600],[200,600],[300,150],[100,100]])
    #pts2 = np.float32([[160,479],[480,479],[400,240],[240,240]])
    pts2 = np.float32([[100,600],[200,600],[250,250],[150,200]])
    psp_mat = cv2.getPerspectiveTransform(pts1,pts2)
    img_dst9 = cv2.warpPerspective(img_src9,psp_mat,size)
    imgshow(img_dst9,"9")

def kadai_01():
    file_name = "kadai_01.JPG"
    kadainum = 1
    img_src = cv2.imread(file_name)#kadai01.JPG=4608:3456=4:3
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_RGB2GRAY)#グレースケール化
    ret,img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)#大津の二値化ret:閾値
    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)#8近傍
    img_close5 = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, element8, iterations = 5)#クロージング
    img_bin = cv2.morphologyEx(img_close5, cv2.MORPH_OPEN, element8, iterations = 5)#オープニング
    height, width, ch = img_src.shape#rows:縦画素数, cols:横画素数, ch:チャンネル数
    #print(f'rows={rows},cols={cols},ch={ch}')

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )#輪郭検出contours:輪郭

    for num in range(len(contours)):
        #num = 0
        cv2.drawContours(img_src, contours, num, (0, 0, 255), 2)#B:0, G:0, R:255輪郭描画 
        #cv2.fillPoly(img_src, contours, (255, 0, 0))#輪郭の中身塗りつぶす
        con = contours[num]#num番の輪郭について
    
        #print(f'輪郭{num}=\n{con_7}')
        #print(f'\n輪郭数={len(contours)}')
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
        print(f'x_max={x_max}')
        print(f'x_min={x_min}')
        print(f'y_max={y_max}')
        print(f'y_min={y_min}')
        cv2.rectangle(img_src,(x_max, y_max),(x_min, y_min),(0,255,0),thickness = 5)#外接矩形
        tate = x_max - x_min
        yoko = y_max - y_min
        print(f'たて:{tate}, よこ:{yoko}')
        area = tate*yoko
        print(f'area={area}')
        text=f'No.{num}, area:{area}'
        cv2.putText(img_src, text, (x_min,y_min-7),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255,0), thickness = 2)
        #img_src=cv2.rectangle(img_src,(2000, 2000),(1000, 1000),(0,255,0))
        #cv2.line(img_src,(x_min,0),(x_min,3456),(0,255,0))
        #cv2.line(img_src,(x_min,y_min),(x_max,y_min),(0,255,0))
        #cv2.line(img_src,(x_max,y_min),(x_max,y_max),(0,255,0))
        #cv2.line(img_src,(x_max,y_max),(x_min,y_max),(0,255,0))
        #cv2.line(img_src,(x_min,y_max),(x_min,y_min),(0,255,0))

        ex1.save(img_src,f'kadai_{kadainum}/kukei4')

        #cv2.line(img_src,(x_max,0),(x_max,3456),(0,255,0))





    shape = img_src.shape#img_srcの配列の型
    x=img_src[0][0]
    #print(f'x={x}') 
    #rint(shape)
    #imgshow(img_src,"src")
    #imgshow(img_otsu,"otsu")
    #imgshow(img_bin,"binary")
    #imgshow(img_src,"saiga")
    
def imgshow(img,a):
    img = resize1(img)
    cv2.imshow(f'{a}',img)#winname,mat

def resize1(img):
    i = 2
    x=400*i
    y=300*i
    img=cv2.resize(img,(x,y))
    return img

if __name__=='__main__':
    main()    