#データの拡張
#cd C:\Users\toneg\coin\opencv
#python expansion.py
import cv2
import os

def main():
    shurui = [1, 5, 10, 50, 100, 500]
    all = 0
    for i in shurui:
        file_cnt = 0
        dir = f'test/data/{i}'
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                file_cnt += 1
        print(f'{i}:{file_cnt}コ')
        all = all + file_cnt
        #kakucho(i, file_cnt)
    print(f'all={all}')

def kakucho(shurui, file_cnt):
    #print(f'shurui={shurui}, file_cnt={file_cnt}')
    for i in range(file_cnt):
        img = cv2.imread(f'test/data/{shurui}/{i}.jpg')
        height, width, ch = img.shape
        #print(f'height={height}, width={width}, ch={ch}')
        center = (int(width/2), int(height/2))
        for j in range(11):
            angle = 30 * (j+1)
            scale = 1
            trans = cv2.getRotationMatrix2D(center, angle , scale)
            image2 = cv2.warpAffine(img, trans, (width,height))
            cv2.imwrite(f'test/data/{shurui}/{i}_ex{j}.jpg',image2)

if __name__ == '__main__':
    main()