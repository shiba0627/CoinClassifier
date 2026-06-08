# CoinClassifier — YOLOv8 硬貨認識プロジェクト

**YOLOv8** を用いて日本国硬貨（1円、5円、10円、50円、100円、500円）を認識するプロジェクトです。
**画像分類 (Classification)** と **物体検出 (Object Detection)** の両方に対応しています。

---

## 📂 プロジェクト構成

```
coinyolo/
├── classification/              # 画像分類
│   ├── train.py                 #   学習スクリプト
│   ├── infer.py                 #   単一画像の推論テスト用スクリプト
│   ├── detect_coins.py          #   Hough変換による硬貨切り抜きとYOLO分類を組み合わせた合計金額計算スクリプト
│   ├── evaluate.py              #   推論結果（Classification/Detection）と正解データを比較し精度評価（TP, FP, FN等）を行うスクリプト
│   └── data/                    #   データセット (train/val)
│
├── detection/                   # 物体検出
│   ├── train.py                 #   学習スクリプト
│   ├── annotate.py              #   アノテーションGUIツール（YOLO形式のBBoxを作成）
│   ├── results_test.py          #   テスト画像に対してYOLO物体検出モデルで推論を行うスクリプト
│   └── data/                    #   データセット
│       ├── raw/                 #     元画像
│       ├── annotations/         #     YOLOラベル (.txt)
│       └── dataset/             #     学習用 (自動生成)
│
├── images/                      # 共有画像
│   ├── test/                    #   テスト用画像 (kadai_01〜10)
│   └── results/                 #   推論・評価結果出力先
│       ├── classification/      #     分類モデルの通常推論結果
│       ├── detection/           #     検出モデルの通常推論結果
│       ├── evaluation_classification/ # 分類モデルの評価指標付き結果画像
│       └── evaluation_detection/      # 検出モデルの評価指標付き結果画像
│
├── models/                      # ベースモデル (.pt)
├── runs/                        # 学習結果 (自動生成)
├── scripts/                     # 過去のスクリプト集
├── check_labels.py              # アノテーション結果（BBox）が正しく画像にマッピングされるか確認するデバッグ用スクリプト
├── check_pil.py                 # PIL(Pillow)の動作確認用スクリプト
└── README.md
```

> **全スクリプトはプロジェクトルートから実行してください。**

---

## 📜 各スクリプトの詳細

### Classification (画像分類) 用スクリプト
- **`classification/train.py`**
  - 用途: YOLOv8の分類モデル（`yolov8n-cls.pt`）を用いて、切り抜かれた硬貨画像の学習を行います。
- **`classification/infer.py`**
  - 用途: 学習済みの分類モデルを用いて、画像1枚がどの硬貨に該当するかを推論テストします。
- **`classification/detect_coins.py`**
  - 用途: テスト画像全体から `cv2.HoughCircles`（OpenCV）を用いて円を検出し、切り抜いた各領域に対して分類モデルを適用して、画像全体の合計金額を算出します。
- **`classification/evaluate.py`**
  - 用途: `detect_coins.py` (Classification方式) および `results_test.py` (Detection方式) の推論結果と、ユーザーが作成した正解データ（Ground Truth）のBBoxとの IoU (Intersection over Union) を計算し、**Precision, Recall, F1-Score** などの評価指標を算出・比較します。結果は画像ファイルおよびターミナル出力に保存されます。

### Detection (物体検出) 用スクリプト
- **`detection/train.py`**
  - 用途: YOLOv8の物体検出モデル（`yolov8n.pt`）を用いて、画像内の硬貨の「位置（バウンディングボックス）」と「種類（クラス）」を同時に検出する学習を行います。
- **`detection/annotate.py`**
  - 用途: GUI上で画像に矩形を描画し、物体検出の学習および評価に必要なYOLO形式のラベルデータ（`.txt`）を作成・保存するツールです。自動検出（Auto Labeling）機能も備えています。
- **`detection/results_test.py`**
  - 用途: 学習済みの物体検出モデルを用いて、テスト画像群に対して一括で推論を行い、バウンディングボックスを描画した結果画像を保存します。

### ルートディレクトリの便利スクリプト
- **`check_labels.py`**
  - 用途: 作成したアノテーションファイル（YOLO形式）を読み込み、元の画像上に正しくバウンディングボックスが描画されるかを確認するための検証用スクリプトです。
- **`check_pil.py`**
  - 用途: Python Imaging Library (Pillow) が正しく動作しているかを確認するための簡易スクリプトです。

> **全スクリプトはプロジェクトルートから実行してください。**

---

## 🚀 使い方

### 📦 `uv` パッケージマネージャーについて（初心者向け）

本プロジェクトでは、Pythonのパッケージ管理・実行ツールとして **[uv](https://docs.astral.sh/uv/)** を使用しています。
`uv` は従来の `pip` や `venv` よりも**非常に高速**で、仮想環境の構築からライブラリのインストールまでを自動で行ってくれます。

> **実行時のルール:**
> すべてのPythonスクリプトは `python ...` ではなく、**`uv run python ...`** という形式で実行します。
> これにより、手動で仮想環境を有効化（activate）しなくても、プロジェクト専用のクリーンな環境で安全にスクリプトを実行できます。

### 1. 環境構築

初回のみ、以下のコマンドを実行して必要なライブラリ（OpenCVやYOLOなど）を自動でインストールします。

```bash
uv sync
```

### 画像分類 (Classification)

1枚の画像に写った硬貨の種類を判定します。

```bash
# 学習
uv run python classification/train.py

# 推論テスト
uv run python classification/infer.py

# 応用: 複数硬貨の合計金額計算
uv run python classification/detect_coins.py
```

### 物体検出 (Object Detection)

1枚の画像内の複数硬貨の位置と種類を同時に検出します。

```bash
# アノテーション (BBox付与ツール)
uv run python detection/annotate.py

# 学習
uv run python detection/train.py
```

---

## 🔰 YOLO初心者向けの基礎知識

| 方式 | モデル | 用途 | 学習データの準備 |
|---|---|---|---|
| **Classification** | `yolov8n-cls.pt` | 画像1枚 → 種類を1つ判定 | フォルダ分けするだけ（簡単） |
| **Detection** | `yolov8n.pt` | 画像1枚 → 複数物体の位置と種類を検出 | BBoxアノテーションが必要 |

---

## 📊 推論精度の比較と可視化 (Classification vs Detection)

評価用スクリプト (`evaluate.py`) を用いて算出した精度評価の結果と可視化例です。
以下のように、Classification（Hough変換+分類）と Detection（YOLO物体検出）で特徴が分かれます。

| 手法 | TP | FP | FN | Precision | Recall | F1-Score | Det. Rate | 特徴・傾向 |
|---|---|---|---|---|---|---|---|---|
| **Classification** | 65 | 20 | 14 | 0.765 | **0.823** | **0.793** | 94.9% (75/79) | 円検出を行うため、硬貨の「見逃し (FN)」が少なく Recall が高い。 |
| **Detection** | 58 | 29 | 21 | 0.667 | 0.734 | 0.699 | **97.5%** (77/79) | 「硬貨の発見」は得意だが、種類の誤認でFN/FPが増えF1が落ちる。 |

### 📈 評価指標（メトリクス）の定義

各指標は以下の定義と計算式に基づいて算出されています。
本プロジェクトでは「推論した枠と正解の枠の重なり(IoU)が50%以上」かつ「金額クラスが一致」した場合を正解（TP）としています。

- **TP (True Positive / 真陽性)**
  - **定義:** 正しい位置に、正しい金額の硬貨を検出できた数。
- **FP (False Positive / 偽陽性)**
  - **定義:** 硬貨がない場所を検出してしまった、または「金額を間違えて」検出してしまった数。（誤検出）
- **FN (False Negative / 偽陰性)**
  - **定義:** 本当は硬貨があるのに、見つけられなかった（または金額を間違えてしまった）数。（見逃し）
- **Precision (適合率)**
  - **定義:** モデルが予測した硬貨のうち、実際に正解だった割合。
  - **計算式:** `TP / (TP + FP)`
  - **意味:** この値が低いと、背景などを誤認している（FPが多い）ことを示します。
- **Recall (再現率)**
  - **定義:** 実際に存在する全硬貨のうち、モデルが正しく見つけ出せた割合。
  - **計算式:** `TP / (TP + FN)`
  - **意味:** この値が低いと、硬貨を見逃している（FNが多い）ことを示します。
- **F1-Score (F値)**
  - **定義:** Precision と Recall の調和平均。モデルの総合的な性能を示す指標。
  - **計算式:** `2 * (Precision * Recall) / (Precision + Recall)`
- **Det. Rate (硬貨検出率 / Coin Detection Rate)**
  - **定義:** 金額（クラス）は問わず、純粋に「硬貨の場所（枠）」を見つけられた割合。
  - **計算式:** `(枠の位置が一致した正解数) / (全正解数)`
  - **意味:** 「物体を発見する能力」のみを比較するための本プロジェクト独自の指標。

### 出力画像の比較例

出力された画像は `./images/results/` 内の各ディレクトリに保存されます。

| 画像 | Classification (Hough変換 + 分類) | Detection (YOLO物体検出) |
|:---:|:---:|:---:|
| **kadai_01** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_01.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_01.jpg) |
| **kadai_02** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_02.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_02.jpg) |
| **kadai_03** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_03.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_03.jpg) |
| **kadai_04** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_04.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_04.jpg) |
| **kadai_05** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_05.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_05.jpg) |
| **kadai_06** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_06.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_06.jpg) |
| **kadai_07** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_07.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_07.jpg) |
| **kadai_08** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_08.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_08.jpg) |
| **kadai_09** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_09.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_09.jpg) |
| **kadai_10** | ![Cls](./images/results/evaluation_classification/result_cls_kadai_10.jpg) | ![Det](./images/results/evaluation_detection/result_det_kadai_10.jpg) |

---

- **Q. Macで学習が遅い / 警告が出る**
  - Apple Silicon (M1/M2/M3/M4) をお使いの場合は `device='mps'` を指定してGPU高速化が可能です（設定済み）。
- **Q. detect_coins.py で硬貨を取りこぼす**
  - `cv2.HoughCircles` のパラメータ（`param1`, `param2`, `minRadius` など）を調整してください。

