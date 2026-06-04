from ultralytics import YOLO

def main():
    # 学習済みのベストモデルを読み込む
    model = YOLO('runs/classify/train/weights/best.pt')

    # テスト対象の画像パス
    image_path = '推論/kadai_01.JPG'

    # 推論（識別）の実行
    print(f"Running inference on {image_path}...")
    results = model(image_path)

    # 結果の表示
    for result in results:
        # result.probs に分類の確率が入っています
        top1_index = result.probs.top1
        top1_class = result.names[top1_index]
        top1_conf = result.probs.top1conf.item()

        print("\n=== 推論結果 ===")
        print(f"予測された硬貨: {top1_class}円")
        print(f"確信度 (Confidence): {top1_conf:.2%}")
        print("===============\n")

if __name__ == "__main__":
    main()
