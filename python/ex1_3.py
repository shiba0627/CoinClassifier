import cv2
import ex1

# 8ビット1チャンネルのグレースケールとして画像を読み込む
img = cv2.imread("kadai_1/5_5.jpg", cv2.IMREAD_GRAYSCALE) 

contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE ) 

# 画像表示用に入力画像をカラーデータに変換する
img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 全ての輪郭を描画
cv2.drawContours(img_disp, contours, -1, (0, 0, 255), 2)

# 輪郭の点の描画
for contour in contours:
    for point in contour:
        cv2.circle(img_disp, point[0], 3, (0, 255, 0), -1)

ex1.save(img_disp,f'kadai_1/rinkaku')
ex1.imgshow(img_disp, "img")#表示
# キー入力待ち(ここで画像が表示される)
cv2.waitKey()