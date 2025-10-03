#cd C:\Users\toneg\coin\opencv
#python make_model.py
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


def main():
    print(tf.__version__)
    classes = sorted(os.listdir('dataset/train'), key = int)
    train_list = []
    valid_list = []
    for i in classes:
        file_cnt = 0
        dir = f'dataset/train/{i}'
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                file_cnt += 1
        print(f'{i}:{file_cnt}コ')
        for j in range(file_cnt):
            train_list.append(f'dataset/train/{i}/_{i}_{j+1}.jpg')
    
    for i in classes:
        file_cnt = 0
        dir = f'dataset/valid/{i}'
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                file_cnt += 1
        print(f'{i}:{file_cnt}コ')
        for j in range(file_cnt):
            valid_list.append(f'dataset/valid/{i}/_{i}_{j+1}.jpg')
    
    train_ds = tf.data.Dataset.list_files(train_list)
    valid_ds = tf.data.Dataset.list_files(valid_list)
    AUTOTUNE = tf.data.experimental.AUTOTUNE # 処理を最適化するためのおまじない（自動チューニング設定）
    train_ds = train_ds.shuffle(len(train_list)) # 訓練データをシャッフルする。引数にはデータ数を指定すると完全なシャッフルが行われる。len(x_train)は60000。
    train_ds = train_ds.repeat(1) # 1 epochで使われるデータの回数。1の場合，1epochで1回しか使われない。引数を空欄にすると無限に使われる。
    train_ds = train_ds.batch(32) # ミニバッチを作る。1バッチ32個のデータ。
    train_ds = train_ds.map(lambda files: tf.py_function(load_file, [files], Tout=[tf.float32, tf.float32])) # ファイル名から入力ラベルとラベルを取得
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # 訓練中に次のバッチを取り出すための処理。

    valid_ds = valid_ds.batch(32) # 検証データはシャッフルする必要ないので，バッチ化のみの処理でOK。
    valid_ds = valid_ds.map(lambda x: tf.py_function(load_file, [x], Tout=[tf.float32, tf.float32]))
    #print(train_list)
    #print(f'valid_ds={valid_ds.shape}')


    cv2.waitKey()#delay[msec]


def load_file(files):
    ys = [] # ラベル
    xs = [] # 入力データ
    for f in files:
        file = bytes.decode(f.numpy()) # ファイル名はTensor型で保存されているため，文字列型として取得する。
        m = re.search(r'_(\d+)_', file) # ちょっと違うパターンの書き方
        label = m.groups()[0]
        ys.append(label) # オーム値をそのままラベルにする（インデックス化しない）
        # print(label, label_id)
        img = cv2.imread(file) # 画像ファイルをカラーで取得
        xs.append(img) # データを入力データリストに追加
    xs = np.array(xs, dtype=np.float32) / 255. # 正規化してfloat32の行列に変換する
    ys = np.array(ys, dtype=np.float32) # 回帰モデルなのでラベルもfloat32に変換する
    print(xs.shape, ys.shape)
    return xs, ys


if __name__ == '__main__':
    main()