#cd C:\Users\toneg\coin\opencv
#python dataset.py
import os
import cv2
import ex1

def main():
    classes = sorted(os.listdir('deep/train'), key = int)
    for i in classes:
        file_cnt = 0
        dir = f'deep/train/{i}'
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                file_cnt += 1
        print(f'{i}:{file_cnt}コ')#i:ラベル, file_cnt:ファイル数
        for cnt in range(file_cnt):
            img = cv2.imread(f'deep/train/{i}/{cnt+1}.jpg')
            ex1.save(img,f'dataset/train/{i}/_{i}_{cnt+1}.jpg')
    
    for i in classes:
        file_cnt = 0
        dir = f'deep/valid/{i}'
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                file_cnt += 1
        print(f'{i}:{file_cnt}コ')
        for cnt in range(file_cnt):
            img = cv2.imread(f'deep/valid/{i}/{cnt+1}.jpg')
            ex1.save(img,f'dataset/valid/{i}/_{i}_{cnt+1}.jpg')


if __name__ == '__main__':
    main()