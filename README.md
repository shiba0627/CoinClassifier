# CoinClassifier — YOLOv8 硬貨認識プロジェクト

**YOLOv8** を用いて日本国硬貨（1円、5円、10円、50円、100円、500円）を認識するプロジェクトです。
**画像分類 (Classification)** と **物体検出 (Object Detection)** の両方に対応しています。

## 📑 目次
- [2つのアプローチの比較](#-2つのアプローチの比較)
- [環境構築と使い方](#-環境構築と使い方)
- [プロジェクト構成とスクリプト詳細](#-プロジェクト構成とスクリプト詳細)
- [推論精度の比較と可視化](#-推論精度の比較と可視化-classification-vs-detection)
- [FAQ・トラブルシューティング](#-faqトラブルシューティング)

---

## 🔍 2つのアプローチの比較

本プロジェクトでは、2種類のモデルを用いてアプローチの違いを検証・実装しています。

| 方式 | モデル | 用途 | 学習データの準備 |
|---|---|---|---|
| **Classification (画像分類)** | `yolov8n-cls.pt` | 画像1枚 → 種類を1つ判定 | フォルダ分けするだけ（簡単） |
| **Detection (物体検出)** | `yolov8n.pt` | 画像1枚 → 複数物体の位置と種類を検出 | BBoxアノテーションが必要 |

---

## 🚀 環境構築と使い方

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

### 2. 画像分類 (Classification) の実行

1枚の画像に写った硬貨の種類を判定します。

```bash
# 学習
uv run python classification/train.py

# 推論テスト
uv run python classification/infer.py

# 応用: 複数硬貨の合計金額計算
uv run python classification/detect_coins.py
```

### 3. 物体検出 (Object Detection) の実行

1枚の画像内の複数硬貨の位置と種類を同時に検出します。

```bash
# アノテーション (BBox付与ツール)
uv run python detection/annotate.py

# 学習
uv run python detection/train.py
```

---

## 📂 プロジェクト構成とスクリプト詳細

### ディレクトリ構成

```text
coinyolo/
├── classification/              # 画像分類
│   ├── train.py                 #   学習スクリプト
│   ├── infer.py                 #   単一画像の推論テスト用スクリプト
│   ├── detect_coins.py          #   Hough変換による硬貨切り抜きとYOLO分類を組み合わせた合計金額計算スクリプト
│   └── data/                    #   データセット (train/val)
│
├── evaluation/                  # 評価
│   └── evaluate.py              #   推論結果（Classification/Detection）と正解データを比較し精度評価を行うスクリプト
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

### 各スクリプトの詳細

#### Classification (画像分類) 用スクリプト
- **`classification/train.py`**
  - 用途: YOLOv8の分類モデル（`yolov8n-cls.pt`）を用いて、切り抜かれた硬貨画像の学習を行います。
- **`classification/infer.py`**
  - 用途: 学習済みの分類モデルを用いて、画像1枚がどの硬貨に該当するかを推論テストします。
- **`classification/detect_coins.py`**
  - 用途: テスト画像全体から `cv2.HoughCircles`（OpenCV）を用いて円を検出し、切り抜いた各領域に対して分類モデルを適用して、画像全体の合計金額を算出します。
- **`evaluation/evaluate.py`**
  - 用途: `detect_coins.py` (Classification方式) および `results_test.py` (Detection方式) の推論結果と、ユーザーが作成した正解データ（Ground Truth）のBBoxとの IoU (Intersection over Union) を計算し、**Precision, Recall, F1-Score** などの評価指標を算出・比較します。結果は画像ファイルおよびターミナル出力に保存されます。

#### Detection (物体検出) 用スクリプト
- **`detection/train.py`**
  - 用途: YOLOv8の物体検出モデル（`yolov8n.pt`）を用いて、画像内の硬貨の「位置（バウンディングボックス）」と「種類（クラス）」を同時に検出する学習を行います。
- **`detection/annotate.py`**
  - 用途: GUI上で画像に矩形を描画し、物体検出の学習および評価に必要なYOLO形式のラベルデータ（`.txt`）を作成・保存するツールです。自動検出（Auto Labeling）機能も備えています。
- **`detection/results_test.py`**
  - 用途: 学習済みの物体検出モデルを用いて、テスト画像群に対して一括で推論を行い、バウンディングボックスを描画した結果画像を保存します。

#### ルートディレクトリの便利スクリプト
- **`check_labels.py`**
  - 用途: 作成したアノテーションファイル（YOLO形式）を読み込み、元の画像上に正しくバウンディングボックスが描画されるかを確認するための検証用スクリプトです。
- **`check_pil.py`**
  - 用途: Python Imaging Library (Pillow) が正しく動作しているかを確認するための簡易スクリプトです。

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
本プロジェクトでは、IoU（Intersection over Union）および クラス一致 を基準として正解判定を行っています。

#### 1. 基本判定基準
- **IoU (Intersection over Union):** 予測バウンディングボックス（BBox）と正解BBoxの重なり具合を示す指標。
  - 計算式: `IoU = (予測BBox ∩ 正解BBox の面積) / (予測BBox ∪ 正解BBox の面積)`
  - 本スクリプトでの閾値: `IoU >= 0.5`
- **クラスの一致:** 予測された硬貨の種類（1円、500円など）が正解データと完全に一致していること。

#### 2. 評価変数の定義
- **TP (True Positive / 真陽性)**
  - **定義:** `IoU >= 0.5` かつ `予測クラス == 正解クラス` を満たした検出数。
- **FP (False Positive / 偽陽性)**
  - **定義:** 以下のいずれかに該当する誤検出数。
    - `IoU >= 0.5` を満たすが `予測クラス != 正解クラス` の場合（金額の誤認）。
    - どの正解BBoxとも `IoU >= 0.5` を満たさない予測（背景の誤認など）。
- **FN (False Negative / 偽陰性)**
  - **定義:** 以下のいずれかに該当する見逃し数。
    - いずれの予測BBoxとも `IoU >= 0.5` を満たさない正解データ（完全な見逃し）。
    - `IoU >= 0.5` を満たす予測BBoxは存在するが、`予測クラス != 正解クラス` の場合（モデルが金額を間違えたため、その金額の正解としては見逃し扱いとなる）。

#### 3. 総合指標の計算式
- **Precision (適合率)**
  - **計算式:** `TP / (TP + FP)`
  - **意味:** モデルが出力した全予測のうち、実際に正解の条件を満たした割合。FP（背景の誤認や金額の誤認）が増えると低下する。
- **Recall (再現率)**
  - **計算式:** `TP / (TP + FN)`
  - **意味:** 実際の全正解データのうち、モデルが正しく見つけ出せた割合。FN（完全な見逃しや金額の誤認）が増えると低下する。
- **F1-Score (F値)**
  - **計算式:** `2 * (Precision * Recall) / (Precision + Recall)`
  - **意味:** Precision と Recall の調和平均。トレードオフの関係にある両者を加味したモデルの総合的な性能指標。
- **Det. Rate (硬貨検出率 / Coin Detection Rate)**
  - **計算式:** `(少なくとも1つの予測BBoxと IoU >= 0.5 を満たした正解BBoxの数) / (全正解BBoxの数)`
  - **意味:** 金額（クラス）の正解・不正解を問わず、純粋に「物体として枠で捉えることができた割合」を示す本プロジェクト独自の指標。

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

## ❓ FAQ・トラブルシューティング

- **Q. Macで学習が遅い / 警告が出る**
  - Apple Silicon (M1/M2/M3/M4) をお使いの場合は `device='mps'` を指定してGPU高速化が可能です（設定済み）。
- **Q. detect_coins.py で硬貨を取りこぼす**
  - `cv2.HoughCircles` のパラメータ（`param1`, `param2`, `minRadius` など）を調整してください。
