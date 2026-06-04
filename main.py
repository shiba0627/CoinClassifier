from ultralytics import YOLO

def main():
    print("Initializing YOLO Classification Training...")
    
    # モデルの読み込み (YOLOv8 の分類用軽量モデルを初期値として設定)
    model = YOLO('yolov8n-cls.pt')
    
    # 学習の実行
    print("Starting training on Japanese coins dataset...")
    results = model.train(
        data='./data', # ローカルのデータセットパス
        epochs=10, 
        imgsz=224,     # YOLO分類モデルのデフォルトサイズ推奨
        device='mps'   # Apple Silicon Macの場合は MPS を有効化
    )
    print("Training finished.")

if __name__ == "__main__":
    main()
