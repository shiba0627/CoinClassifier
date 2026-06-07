from ultralytics import YOLO
import os

def main():
    # モデルの読み込み
    model_path = "runs/detect/coin_detect/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"エラー: モデルが見つかりません: {model_path}")
        return
        
    model = YOLO(model_path)
    
    # 出力先ディレクトリ (プロジェクト構成に合わせて images/results/detection に統一)
    output_dir = "images/results/detection"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"推論結果の保存先: {output_dir}")
    print("-" * 40)
    
    # kadai_01 から kadai_10 まで推論
    for i in range(1, 11):
        base_name = f"kadai_{i:02d}"
        img_path = None
        
        # 拡張子が大文字小文字両方に対応
        if os.path.exists(f"images/test/{base_name}.JPG"):
            img_path = f"images/test/{base_name}.JPG"
        elif os.path.exists(f"images/test/{base_name}.jpg"):
            img_path = f"images/test/{base_name}.jpg"
            
        if not img_path:
            print(f"スキップ: {base_name}.JPG が images/test/ に見つかりません")
            continue
            
        print(f"推論中: {base_name} ...", end=" ")
        
        results = model(img_path, conf=0.3, verbose=False)
        
        for r in results:
            out_path = os.path.join(output_dir, f"result_{base_name}.jpg")
            r.save(out_path)
            print(f"検出数: {len(r.boxes)} -> 保存完了")

if __name__ == "__main__":
    main()