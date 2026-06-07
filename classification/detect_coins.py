import cv2
import numpy as np
import glob
import os
from ultralytics import YOLO

def process_coins(image_path, model, output_path='result_output.jpg'):
    print(f"--- 処理開始: {image_path} ---")
    
    img = cv2.imread(image_path)
    if img is None:
        print("画像が読み込めませんでした。パスを確認してください。")
        return
        
    # リサイズ (処理速度向上とノイズ削減)
    height, width = img.shape[:2]
    max_size = 1024
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ハフ変換にはメディアンフィルタがノイズ除去に効果的
    blurred = cv2.medianBlur(gray, 7) 
    
    # ハフ変換による高精度な円検出
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50, 
        param1=50, 
        param2=30, 
        minRadius=15, 
        maxRadius=120
    )
    
    total_amount = 0
    coin_count = 0
    result_img = img.copy()
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            
            # 円の半径を少し広げて確実に硬貨全体が入るようにする
            pad = int(r * 0.15)
            r_padded = r + pad
            
            # 画像の境界を超えないように切り抜き範囲を計算
            x1 = max(0, int(x - r_padded))
            y1 = max(0, int(y - r_padded))
            x2 = min(img.shape[1], int(x + r_padded))
            y2 = min(img.shape[0], int(y + r_padded))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            coin_crop = img[y1:y2, x1:x2]
            if coin_crop.shape[0] == 0 or coin_crop.shape[1] == 0:
                continue
            
            # YOLOで切り抜いた硬貨を推論
            results = model(coin_crop, verbose=False)
            
            for res in results:
                top1_class = res.names[res.probs.top1]
                conf = res.probs.top1conf.item()
                
                if conf > 0.4:
                    try:
                        amount = int(top1_class)
                        total_amount += amount
                        coin_count += 1
                        
                        # 円形を描画 (緑色)
                        cv2.circle(result_img, (x, y), r, (0, 255, 0), 2)
                        # 中心点を描画 (赤色)
                        cv2.circle(result_img, (x, y), 2, (0, 0, 255), 3)
                        
                        # テキストを描画
                        label = f"{amount}Yen({conf:.2f})"
                        cv2.putText(result_img, label, (x - r, max(y - r - 10, 0)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    except ValueError:
                        pass
                        
    # 合計金額を描画
    cv2.putText(result_img, f"Total: {total_amount} Yen ({coin_count} coins)", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                
    # 保存
    cv2.imwrite(output_path, result_img)
    print(f"処理完了: {coin_count}枚の硬貨を検出しました。合計金額: {total_amount}円")
    print(f"結果画像を保存しました: {output_path}\n")

if __name__ == "__main__":
    # モデルのロードをループの外で行う（高速化のため）
    model_path = 'runs/classify/train/weights/best.pt'  # プロジェクトルートから実行
    print(f"モデルをロード中: {model_path}")
    loaded_model = YOLO(model_path)
    
    # kadai_01 から kadai_10 までを一括処理
    for i in range(1, 11):
        # 拡張子が大文字小文字どちらの可能性もあるため両方チェック
        base_name = f"kadai_{i:02d}"
        target_file = None
        
        if os.path.exists(f"images/test/{base_name}.JPG"):
            target_file = f"images/test/{base_name}.JPG"
        elif os.path.exists(f"images/test/{base_name}.jpg"):
            target_file = f"images/test/{base_name}.jpg"
            
        if target_file:
            out_file = f"images/results/classification/result_{base_name}.jpg"
            process_coins(target_file, loaded_model, output_path=out_file)
        else:
            print(f"images/test/{base_name}.JPG は見つかりませんでした。")


