import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.regularizers import l2 # L2正則化
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard # TensorBoard用のコールバック
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os # ディレクトリ（フォルダ）やファイルを扱うためのライブラリ（本当はPathlibライブラリのほうが良いが難しいので簡単な方で）
import glob # ファイル一覧を取得するためのライブラリ
import re # 正規表現を使ったパターンマッチング用（ラベルを取得するため）
from keras.preprocessing import image # keras.preprocessing.image APIを利用する。画像拡張用の関数が用意されている。
from keras.preprocessing.image import ImageDataGenerator
tf.test.gpu_device_name() # GPUの利用確認
import eight
R = [1, 5,10,50, 100,500]
RR = []

for i in R:
    #img = cv2.imread('kadai_1/one_kukei4.jpg')
    img = cv2.imread(f'deep/data/{i}/2.jpg')
    plt.imshow(img)
    img = img.reshape(1, 300, 300, 3)

    img = np.float32(img)/255.
    print(f'shape ={img.shape}')
    model = tf.keras.models.load_model('model/takusan_epoch_10.6.h5')
    pred = model.predict(img)
    print(f'{i}pred => {pred}')
    RR.append(pred)
print('-----回帰モデル-----')
for i in range(6):
    print(f'正解ラベル : {R[i]} => 推論結果 : {RR[i]}')

plt.show()

