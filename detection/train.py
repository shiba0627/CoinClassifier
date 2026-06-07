"""
YOLOv8 物体検出 (Object Detection) 訓練スクリプト

使い方:
    uv run python detection/train.py

処理の流れ:
    1. detection/data/raw/ の画像と detection/data/annotations/ のラベルを
       YOLO形式のフォルダ構造 (images/train, images/val, labels/train, labels/val) に整理
    2. data.yaml を生成
    3. YOLOv8n で物体検出モデルを学習
"""

import os
import sys
import shutil
import random
import yaml
from pathlib import Path

# ============================================================
# 設定
# ============================================================

# パス設定
IMAGE_SOURCE_DIR = "detection/data/raw"              # アノテーション済み画像の元フォルダ
LABEL_SOURCE_DIR = "detection/data/annotations"       # アノテーションファイルのフォルダ
DATASET_DIR = "detection/data/dataset"                 # YOLO用データセット出力先
BASE_MODEL = "models/yolov8s.pt"                       # ベースモデル (物体検出用)

# 学習パラメータ
EPOCHS = 300
IMAGE_SIZE = 640
BATCH_SIZE = 8          # 画像枚数が少ないので小さめ
DEVICE = "mps"          # Apple Silicon Mac
VAL_SPLIT = 0.2         # 検証用データの割合 (20%)

# クラス定義
CLASS_NAMES = ["1", "5", "10", "50", "100", "500"]

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


# ============================================================
# データセット準備
# ============================================================

def prepare_dataset():
    """画像とラベルをYOLO形式のディレクトリ構造に整理する"""
    print("=" * 60)
    print("📁 データセットの準備")
    print("=" * 60)

    # 出力ディレクトリを作成
    dirs = [
        f"{DATASET_DIR}/images/train",
        f"{DATASET_DIR}/images/val",
        f"{DATASET_DIR}/labels/train",
        f"{DATASET_DIR}/labels/val",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 画像ファイルの一覧を取得
    image_files = sorted([
        f for f in os.listdir(IMAGE_SOURCE_DIR)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])

    # 対応するラベルファイルが存在するもののみを対象にする
    paired_files = []
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        label_file = base + ".txt"
        label_path = os.path.join(LABEL_SOURCE_DIR, label_file)
        if os.path.exists(label_path):
            # ラベルファイルが空でないかチェック
            if os.path.getsize(label_path) > 0:
                paired_files.append((img_file, label_file))
            else:
                print(f"  ⚠ スキップ (空のラベル): {img_file}")
        else:
            print(f"  ⚠ スキップ (ラベルなし): {img_file}")

    if not paired_files:
        print("❌ アノテーション済みの画像が見つかりません。")
        print(f"   画像フォルダ: {IMAGE_SOURCE_DIR}")
        print(f"   ラベルフォルダ: {LABEL_SOURCE_DIR}")
        sys.exit(1)

    print(f"\n  📊 アノテーション済み画像: {len(paired_files)} 枚")

    # train / val に分割
    random.seed(42)
    random.shuffle(paired_files)
    val_count = max(1, int(len(paired_files) * VAL_SPLIT))
    val_files = paired_files[:val_count]
    train_files = paired_files[val_count:]

    # 画像が少ない場合は trainにも valの画像を含める (最低限の学習データ確保)
    if len(train_files) < 2:
        print("  ⚠ 学習データが少ないため、検証データも学習に使用します")
        train_files = paired_files
        val_files = paired_files[:val_count]

    print(f"  🏋️ 学習用: {len(train_files)} 枚")
    print(f"  🧪 検証用: {len(val_files)} 枚\n")

    # ファイルをコピー
    def copy_files(file_pairs, split_name):
        for img_file, label_file in file_pairs:
            # 画像をコピー
            src_img = os.path.join(IMAGE_SOURCE_DIR, img_file)
            dst_img = os.path.join(DATASET_DIR, "images", split_name, img_file)
            shutil.copy2(src_img, dst_img)

            # ラベルをコピー
            src_label = os.path.join(LABEL_SOURCE_DIR, label_file)
            dst_label = os.path.join(DATASET_DIR, "labels", split_name, label_file)
            shutil.copy2(src_label, dst_label)

        print(f"  ✅ {split_name}: {len(file_pairs)} ファイルをコピーしました")

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    # data.yaml を生成
    dataset_abs_path = os.path.abspath(DATASET_DIR)
    data_config = {
        "path": dataset_abs_path,
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  📄 data.yaml を生成しました: {yaml_path}")
    print(f"     path: {dataset_abs_path}")
    print(f"     classes: {CLASS_NAMES}")

    # アノテーション統計を表示
    print_annotation_stats(train_files + val_files)

    return yaml_path


def print_annotation_stats(file_pairs):
    """アノテーションの統計情報を表示"""
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    total_boxes = 0

    for _, label_file in file_pairs:
        label_path = os.path.join(LABEL_SOURCE_DIR, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    total_boxes += 1

    print(f"\n  📊 アノテーション統計:")
    print(f"     合計 BBox 数: {total_boxes}")
    for cid, count in sorted(class_counts.items()):
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else "?"
        bar = "█" * min(count, 30)
        print(f"     {name:>4}円: {count:>3} 個 {bar}")


# ============================================================
# 学習
# ============================================================

def train(yaml_path):
    """YOLOv8 物体検出モデルを学習する"""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("🚀 YOLOv8 物体検出モデルの学習開始")
    print("=" * 60)
    print(f"  ベースモデル: {BASE_MODEL}")
    print(f"  データセット: {yaml_path}")
    print(f"  エポック数:   {EPOCHS}")
    print(f"  画像サイズ:   {IMAGE_SIZE}")
    print(f"  バッチサイズ: {BATCH_SIZE}")
    print(f"  デバイス:     {DEVICE}")
    print("=" * 60 + "\n")

    # モデルのロード
    model = YOLO(BASE_MODEL)

    # 学習実行
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=os.path.abspath("runs/detect"),
        name="coin_detect",
        exist_ok=True,
        # データ拡張 (少量データ向けに強めに設定)
        augment=True,
        hsv_h=0.015,       # 色相の変動
        hsv_s=0.7,         # 彩度の変動
        hsv_v=0.4,         # 明度の変動
        degrees=15.0,      # 回転
        translate=0.1,      # 平行移動
        scale=0.5,          # スケール
        flipud=0.5,         # 上下反転
        fliplr=0.5,         # 左右反転
        mosaic=1.0,         # モザイク拡張
    )

    print("\n" + "=" * 60)
    print("✅ 学習完了!")
    print("=" * 60)
    print(f"  重みファイル: runs/detect/coin_detect/weights/best.pt")
    print(f"  結果グラフ:   runs/detect/coin_detect/results.png")
    print(f"\n  推論テスト:")
    print(f"    uv run python -c \"")
    print(f"    from ultralytics import YOLO")
    print(f"    model = YOLO('runs/detect/coin_detect/weights/best.pt')") 
    print(f"    results = model('images/test/kadai_01.JPG')") 
    print(f"    results[0].show()\"")
    print("=" * 60)


# ============================================================
# メイン
# ============================================================

def main():
    print("\n🪙 硬貨物体検出 — YOLOv8 訓練パイプライン\n")

    # 1. データセット準備
    yaml_path = prepare_dataset()

    # 2. 学習
    train(yaml_path)


if __name__ == "__main__":
    main()
