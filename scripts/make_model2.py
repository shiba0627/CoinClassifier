#python make_model2.py
import cv2
import os
import eight
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras import Model

def main():
    x_train = []#データ
    y_train  = []#ラベル
    x_valid = []#データ
    y_valid = []#ラベル
    cnt = 0
    classes, train_cnt, valid_cnt = file_count()
    print(f'{classes[1]}')
    print(f'クラス={classes}, train_cnt={train_cnt}, valid_cnt = {valid_cnt}')
    for i in classes:
        for j in range(train_cnt[cnt]):
            img = cv2.imread(f'deep/train/{i}/{j+1}.jpg')
            x_train.append(img)
            x=int(i)
            y_train.append(x)
        cnt = cnt+1

    cnt = 0
    for i in classes:
        for j in range(valid_cnt[cnt]):
            img = cv2.imread(f'deep/valid/{i}/{j+1}.jpg')
            x_valid.append(img)
            x=int(i)
            y_valid.append(x)

        cnt = cnt+1

    print(f'len(x_train)={len(x_train)}, len(y_train)={len(y_train)}')
    print(f'len(x_valid)={len(x_valid)}, len(y_valid)={len(y_valid)}')
    print(f'x_train[1].shape={x_train[1].shape}')
    #print(y_train)
    #eight.imgshow(train[0],'1')
    cv2.waitKey()#delay[msec]
    print(f'y_train[1]={y_train[1]}')

    x_train = np.float32(x_train)
    x_valid = np.float32(x_valid)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    AUTOTUNE = tf.data.experimental.AUTOTUNE # 処理を最適化するためのおまじない（自動チューニング設定）
    train_ds = train_ds.shuffle(len(x_train)) # 訓練データをシャッフルする。引数にはデータ数を指定すると完全なシャッフルが行われる。len(x_train)は60000。
    train_ds = train_ds.repeat(1) # 1 epochで使われるデータの回数。1の場合，1epochで1回しか使われない。引数を空欄にすると無限に使われる。
    train_ds = train_ds.batch(32) # ミニバッチを作る。1バッチ32個のデータ。
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) # 訓練中に次のバッチを取り出すための処理。
    valid_ds = valid_ds.batch(32) # 検証データはシャッフルする必要ないので，バッチ化のみの処理でOK

    # 今回は Functional API を使ってみる。（今後こっちの書き方でやります）
    input = Input(shape=(300, 300, 3), name='input') # 入力層の定義
    h = Flatten()(input) # 28x28の2次元テンソルを784次元の1次元テンソルに変換する。フラット化。
    h = Dense(512, activation='relu', name='dense1')(h) # 隠れ層のノードは512
    output = Dense(6, activation='softmax', name='output')(h) # 出力層

    model = Model(inputs=input, outputs=output) # この処理でモデルを実体化する。入力層と出力層を渡すと自動的にネットワークを構築してくれる。
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(train_ds, epochs=1, validation_data=valid_ds) # エポック数は10くらいで十分
    batch = next(iter(train_ds))
    print(batch[0].shape)
    print(batch[1].shape)

    cv2.destroyAllWindows()


def file_count():
    train_cnt = []
    valid_cnt =[]
    classes = sorted(os.listdir('deep/train'), key = int)
    for i in classes:
        file_cnt = 0
        dir = f'deep/train/{i}'
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                file_cnt += 1
        #print(f'{i}:{file_cnt}コ')
        train_cnt.append(file_cnt)

    for i in classes:
        file_cnt = 0
        dir = f'deep/valid/{i}'
        for path in os.listdir(dir):
            if os.path.isfile(os.path.join(dir, path)):
                file_cnt += 1
        #print(f'{i}:{file_cnt}コ')
        valid_cnt.append(file_cnt)
    return classes, train_cnt, valid_cnt

if __name__ == '__main__':
    main()

