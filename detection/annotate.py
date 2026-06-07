"""
硬貨アノテーションツール (YOLO Object Detection 用)

使い方:
    uv run python detection/annotate.py
    → 起動時にフォルダ選択ダイアログが表示されます

    uv run python detection/annotate.py [画像ディレクトリ] [出力ディレクトリ]
    → コマンドライン引数でも指定可能

操作方法:
    - マウスドラッグ: バウンディングボックスを描画
    - 右クリック: 直近のバウンディングボックスを削除 (Undo)
    - 数字キー / ボタン: クラスを選択
    - ← → キー: 前/次の画像へ移動
    - Ctrl+S / ⌘+S: 現在の画像のアノテーションを保存
    - Ctrl+Z / ⌘+Z: Undo (直近のバウンディングボックスを削除)

出力形式:
    YOLO形式 (.txt) — 各行: class_id center_x center_y width height (正規化座標)
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageOps
import os
import sys
import json

# ============================================================
# 定数
# ============================================================

# 硬貨クラス定義 (class_id: label)
CLASSES = {
    0: "1",
    1: "5",
    2: "10",
    3: "50",
    4: "100",
    5: "500",
}

# クラスごとの表示色 — macOS ダークモードでも映える鮮やかな色
CLASS_COLORS = {
    0: "#FF4757",   # 1円  - 鮮やかな赤
    1: "#FFA502",   # 5円  - 鮮やかなオレンジ
    2: "#2ED573",   # 10円 - 鮮やかな緑
    3: "#1E90FF",   # 50円 - ドジャーブルー
    4: "#A855F7",   # 100円 - 鮮やかな紫
    5: "#06D6A0",   # 500円 - エメラルド
}

# 選択状態のハイライト色 (各クラスの明るいバージョン)
CLASS_HIGHLIGHT = {
    0: "#FF6B81",
    1: "#FFB830",
    2: "#5AE28C",
    3: "#54AAFF",
    4: "#BF7FF7",
    5: "#34E8B8",
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

CANVAS_MAX_WIDTH = 1000
CANVAS_MAX_HEIGHT = 700

# ============================================================
# テーマカラー — 明るいライトテーマ (macOS ダークモードでも見やすい)
# ============================================================
THEME = {
    "bg":               "#F0F2F5",   # メイン背景 (ライトグレー)
    "toolbar_bg":       "#FFFFFF",   # ツールバー背景
    "panel_bg":         "#FFFFFF",   # サイドパネル背景
    "canvas_bg":        "#E2E8F0",   # キャンバス背景
    "text_primary":     "#1A202C",   # メインテキスト (濃いグレー)
    "text_secondary":   "#4A5568",   # サブテキスト
    "text_muted":       "#718096",   # 薄いテキスト
    "status_bg":        "#2D3748",   # ステータスバー背景
    "status_fg":        "#F7FAFC",   # ステータスバーテキスト
    "border":           "#CBD5E0",   # ボーダー
    "btn_nav_bg":       "#4A5568",   # ナビボタン背景
    "btn_nav_fg":       "#FFFFFF",   # ナビボタンテキスト
    "btn_save_bg":      "#38A169",   # 保存ボタン背景
    "btn_undo_bg":      "#E53E3E",   # Undoボタン背景
    "btn_clear_bg":     "#DD6B20",   # クリアボタン背景
    "btn_saveall_bg":   "#805AD5",   # 全保存ボタン背景
    "btn_folder_bg":    "#3182CE",   # フォルダボタン背景
}


# ============================================================
# アノテーションツール クラス
# ============================================================

class AnnotationTool:
    def __init__(self, root, image_dir, output_dir):
        self.root = root
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 画像一覧を取得
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        if not self.image_files:
            messagebox.showerror("エラー", f"画像が見つかりません: {image_dir}")
            root.destroy()
            return

        self.current_index = 0
        self.current_class = 0
        self.boxes = []  # [(class_id, x1, y1, x2, y2), ...] — キャンバス座標
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.temp_rect = None

        # 表示用
        self.scale = 1.0
        self.pil_image = None
        self.tk_image = None
        self.img_width = 0
        self.img_height = 0

        self._build_ui()
        self._load_image(0)

    # ----------------------------------------------------------
    # UI 構築
    # ----------------------------------------------------------
    def _build_ui(self):
        self.root.title("硬貨アノテーションツール — YOLO Object Detection")
        self.root.configure(bg=THEME["bg"])

        # --- 上部: ツールバー ---
        toolbar = tk.Frame(self.root, bg=THEME["toolbar_bg"], pady=6, padx=10,
                           highlightbackground=THEME["border"], highlightthickness=1)
        toolbar.pack(fill=tk.X)

        # ナビゲーション
        nav_frame = tk.Frame(toolbar, bg=THEME["toolbar_bg"])
        nav_frame.pack(side=tk.LEFT)

        self.btn_prev = tk.Button(nav_frame, text="◀ 前へ", command=self._prev_image,
                                  bg=THEME["btn_nav_bg"], fg=THEME["btn_nav_fg"],
                                  font=("Helvetica", 12),
                                  activebackground="#5A6578", activeforeground="white",
                                  relief=tk.FLAT, padx=12, pady=4)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.btn_next = tk.Button(nav_frame, text="次へ ▶", command=self._next_image,
                                  bg=THEME["btn_nav_bg"], fg=THEME["btn_nav_fg"],
                                  font=("Helvetica", 12),
                                  activebackground="#5A6578", activeforeground="white",
                                  relief=tk.FLAT, padx=12, pady=4)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        # フォルダ変更ボタン
        self.btn_folder = tk.Button(nav_frame, text="📁 フォルダ変更", command=self._change_folder,
                                    bg=THEME["btn_folder_bg"], fg="white",
                                    font=("Helvetica", 11),
                                    activebackground="#4299E1", activeforeground="white",
                                    relief=tk.FLAT, padx=10, pady=4)
        self.btn_folder.pack(side=tk.LEFT, padx=(10, 2))

        self.lbl_file = tk.Label(toolbar, text="", bg=THEME["toolbar_bg"],
                                 fg=THEME["text_primary"],
                                 font=("Helvetica", 13, "bold"))
        self.lbl_file.pack(side=tk.LEFT, padx=20)

        # 自動検出ボタン
        self.btn_auto = tk.Button(toolbar, text="✨ 自動検出 (Auto Label)", command=self._auto_detect,
                                  bg="#F6E05E", fg="#1A202C",
                                  font=("Helvetica", 12, "bold"),
                                  activebackground="#ECC94B", activeforeground="#1A202C",
                                  relief=tk.FLAT, padx=12, pady=4)
        self.btn_auto.pack(side=tk.LEFT, padx=(10, 5))

        # Confスライダー
        self.conf_var = tk.DoubleVar(value=0.15)
        conf_frame = tk.Frame(toolbar, bg=THEME["toolbar_bg"])
        conf_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(conf_frame, text="Conf:", bg=THEME["toolbar_bg"], fg=THEME["text_secondary"], font=("Helvetica", 10)).pack(side=tk.LEFT)
        self.scale_conf = tk.Scale(conf_frame, variable=self.conf_var, from_=0.01, to=1.0, resolution=0.01, 
                                   orient=tk.HORIZONTAL, bg=THEME["toolbar_bg"], highlightthickness=0, length=100)
        self.scale_conf.pack(side=tk.LEFT)

        # 保存ボタン
        self.btn_save = tk.Button(toolbar, text="💾 保存 (⌘S)", command=self._save_annotations,
                                  bg=THEME["btn_save_bg"], fg="white",
                                  font=("Helvetica", 12, "bold"),
                                  activebackground="#48BB78", activeforeground="white",
                                  relief=tk.FLAT, padx=16, pady=4)
        self.btn_save.pack(side=tk.RIGHT, padx=2)

        # 全保存ボタン
        self.btn_save_all = tk.Button(toolbar, text="📦 全画像保存", command=self._save_all,
                                      bg=THEME["btn_saveall_bg"], fg="white",
                                      font=("Helvetica", 12),
                                      activebackground="#9F7AEA", activeforeground="white",
                                      relief=tk.FLAT, padx=12, pady=4)
        self.btn_save_all.pack(side=tk.RIGHT, padx=2)

        # Undoボタン
        self.btn_undo = tk.Button(toolbar, text="↩ Undo (⌘Z)", command=self._undo,
                                  bg=THEME["btn_undo_bg"], fg="white",
                                  font=("Helvetica", 12),
                                  activebackground="#FC8181", activeforeground="white",
                                  relief=tk.FLAT, padx=12, pady=4)
        self.btn_undo.pack(side=tk.RIGHT, padx=2)

        # クリアボタン
        self.btn_clear = tk.Button(toolbar, text="🗑 全消去", command=self._clear_boxes,
                                   bg=THEME["btn_clear_bg"], fg="white",
                                   font=("Helvetica", 12),
                                   activebackground="#ED8936", activeforeground="white",
                                   relief=tk.FLAT, padx=12, pady=4)
        self.btn_clear.pack(side=tk.RIGHT, padx=2)

        # --- 中央: メインエリア ---
        main_frame = tk.Frame(self.root, bg=THEME["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左: クラス選択パネル
        class_panel = tk.Frame(main_frame, bg=THEME["panel_bg"], width=150, padx=10, pady=10,
                               highlightbackground=THEME["border"], highlightthickness=1)
        class_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0), pady=4)
        class_panel.pack_propagate(False)

        tk.Label(class_panel, text="クラス選択", bg=THEME["panel_bg"],
                 fg=THEME["text_primary"],
                 font=("Helvetica", 13, "bold")).pack(pady=(4, 12))

        self.class_buttons = {}
        for cid, label in CLASSES.items():
            color = CLASS_COLORS[cid]
            btn = tk.Button(
                class_panel,
                text=f"  {label}円  ",
                command=lambda c=cid: self._select_class(c),
                bg=color, fg="white",
                font=("Helvetica", 14, "bold"),
                activebackground=CLASS_HIGHLIGHT[cid],
                activeforeground="white",
                relief=tk.FLAT, padx=8, pady=8,
                width=8,
            )
            btn.pack(pady=3, fill=tk.X)
            self.class_buttons[cid] = btn

        # ショートカット説明
        tk.Label(class_panel, text="", bg=THEME["panel_bg"]).pack(pady=4)
        sep = tk.Frame(class_panel, bg=THEME["border"], height=1)
        sep.pack(fill=tk.X, pady=4)

        tk.Label(class_panel, text="ショートカット", bg=THEME["panel_bg"],
                 fg=THEME["text_primary"],
                 font=("Helvetica", 11, "bold")).pack(pady=(6, 4), anchor="w")

        shortcuts = [
            "1: 1円    2: 5円",
            "3: 10円  4: 50円",
            "5: 100円  6: 500円",
            "",
            "← →: 画像移動",
            "ドラッグ: BBox描画",
            "右クリック: Undo",
            "⌘S: 保存",
            "⌘Z: Undo",
        ]
        for s in shortcuts:
            tk.Label(class_panel, text=s, bg=THEME["panel_bg"],
                     fg=THEME["text_muted"],
                     font=("Helvetica", 10), anchor="w").pack(anchor="w")

        # フォルダパス表示
        tk.Label(class_panel, text="", bg=THEME["panel_bg"]).pack(pady=2)
        sep2 = tk.Frame(class_panel, bg=THEME["border"], height=1)
        sep2.pack(fill=tk.X, pady=4)
        self.lbl_folder = tk.Label(class_panel, text=f"📂 画像:\n  {os.path.basename(self.image_dir)}\n\n💾 保存先:\n  {os.path.basename(self.output_dir)}",
                                   bg=THEME["panel_bg"], fg=THEME["text_secondary"],
                                   font=("Helvetica", 10), anchor="w", justify=tk.LEFT, wraplength=130)
        self.lbl_folder.pack(anchor="w", pady=(0, 10))
        self.lbl_imgcount = tk.Label(class_panel, text=f"🖼 {len(self.image_files)} 枚",
                                     bg=THEME["panel_bg"], fg=THEME["text_secondary"],
                                     font=("Helvetica", 10), anchor="w")
        self.lbl_imgcount.pack(anchor="w")

        # 右: キャンバス
        canvas_frame = tk.Frame(main_frame, bg=THEME["canvas_bg"],
                                highlightbackground=THEME["border"], highlightthickness=1)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = tk.Canvas(canvas_frame, bg=THEME["canvas_bg"], cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # キャンバスイベント
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<ButtonPress-2>", self._undo_event)   # 中クリック
        self.canvas.bind("<ButtonPress-3>", self._undo_event)   # 右クリック

        # --- 下部: ステータスバー ---
        self.status_bar = tk.Label(self.root, text="", bg=THEME["status_bg"],
                                   fg=THEME["status_fg"],
                                   font=("Helvetica", 11), anchor="w", padx=10, pady=5)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # キーボードショートカット
        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("<Key-1>", lambda e: self._select_class(0))
        self.root.bind("<Key-2>", lambda e: self._select_class(1))
        self.root.bind("<Key-3>", lambda e: self._select_class(2))
        self.root.bind("<Key-4>", lambda e: self._select_class(3))
        self.root.bind("<Key-5>", lambda e: self._select_class(4))
        self.root.bind("<Key-6>", lambda e: self._select_class(5))
        self.root.bind("<Control-s>", lambda e: self._save_annotations())
        self.root.bind("<Command-s>", lambda e: self._save_annotations())  # Mac
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Command-z>", lambda e: self._undo())  # Mac

        # 初期クラス選択ハイライト
        self._highlight_selected_class()

    # ----------------------------------------------------------
    # 画像ロード
    # ----------------------------------------------------------
    def _load_image(self, index):
        if index < 0 or index >= len(self.image_files):
            return

        # 現在のアノテーションを保存するか確認（変更がある場合）
        if self.boxes and index != self.current_index:
            # 自動保存
            self._save_annotations(quiet=True)

        self.current_index = index
        filepath = os.path.join(self.image_dir, self.image_files[index])

        self.pil_image = Image.open(filepath)
        self.pil_image = ImageOps.exif_transpose(self.pil_image) # EXIFの回転を適用
        self.img_width, self.img_height = self.pil_image.size

        # スケーリング計算
        scale_w = CANVAS_MAX_WIDTH / self.img_width
        scale_h = CANVAS_MAX_HEIGHT / self.img_height
        self.scale = min(scale_w, scale_h, 1.0)

        display_w = int(self.img_width * self.scale)
        display_h = int(self.img_height * self.scale)

        display_image = self.pil_image.resize((display_w, display_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display_image)

        self.canvas.config(width=display_w, height=display_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # 既存のアノテーションをロード
        self.boxes = []
        self._load_existing_annotations()
        self._redraw_boxes()

        # UI更新
        fname = self.image_files[index]
        self.lbl_file.config(text=f"📷 {fname}  ({index + 1}/{len(self.image_files)})")
        self._update_status()

    def _load_existing_annotations(self):
        """既存のYOLOアノテーションファイルがあれば読み込む"""
        fname = self.image_files[self.current_index]
        base = os.path.splitext(fname)[0]
        ann_path = os.path.join(self.output_dir, base + ".txt")

        if not os.path.exists(ann_path):
            return

        with open(ann_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # YOLO正規化座標 → キャンバスピクセル座標に変換
                abs_cx = cx * self.img_width * self.scale
                abs_cy = cy * self.img_height * self.scale
                abs_w = w * self.img_width * self.scale
                abs_h = h * self.img_height * self.scale

                x1 = abs_cx - abs_w / 2
                y1 = abs_cy - abs_h / 2
                x2 = abs_cx + abs_w / 2
                y2 = abs_cy + abs_h / 2

                self.boxes.append((class_id, x1, y1, x2, y2))

    # ----------------------------------------------------------
    # フォルダ変更
    # ----------------------------------------------------------
    def _change_folder(self):
        """フォルダ選択ダイアログを表示して画像フォルダを変更"""
        new_dir = filedialog.askdirectory(
            title="画像フォルダを選択",
            initialdir=self.image_dir
        )
        if not new_dir:
            return

        # 現在の画像のアノテーションを自動保存
        if self.boxes:
            self._save_annotations(quiet=True)

        # 新しいフォルダの画像を検索
        new_files = sorted([
            f for f in os.listdir(new_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        if not new_files:
            messagebox.showwarning("警告", f"画像が見つかりません:\n{new_dir}")
            return

        # フォルダ切り替え
        self.image_dir = new_dir
        self.image_files = new_files
        self.current_index = 0
        self.boxes = []

        # 出力ディレクトリも更新 (画像フォルダと同階層に annotations フォルダを作成)
        parent = os.path.dirname(new_dir)
        self.output_dir = os.path.join(parent, "annotations")
        os.makedirs(self.output_dir, exist_ok=True)

        # UI更新
        self.lbl_folder.config(text=f"📂 {os.path.basename(self.image_dir)}")
        self.lbl_imgcount.config(text=f"🖼 {len(self.image_files)} 枚")
        self.root.title(f"硬貨アノテーションツール — {os.path.basename(new_dir)}")

        self._load_image(0)
        self._update_status(f"📂 フォルダ変更: {os.path.basename(new_dir)} ({len(new_files)} 枚)")

    # ----------------------------------------------------------
    # 描画
    # ----------------------------------------------------------
    def _redraw_boxes(self):
        """全てのバウンディングボックスを再描画"""
        self.canvas.delete("bbox")
        for i, (cid, x1, y1, x2, y2) in enumerate(self.boxes):
            color = CLASS_COLORS.get(cid, "#ffffff")
            # ボックス描画 (太めの線)
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=3, tags="bbox"
            )
            # ラベル描画
            label = f"{CLASSES.get(cid, '?')}円"
            # 背景付きテキスト
            text_id = self.canvas.create_text(
                x1 + 3, y1 - 3,
                text=label, fill="white", anchor=tk.SW,
                font=("Helvetica", 12, "bold"), tags="bbox"
            )
            bbox = self.canvas.bbox(text_id)
            if bbox:
                pad = 3
                self.canvas.create_rectangle(
                    bbox[0] - pad, bbox[1] - pad,
                    bbox[2] + pad, bbox[3] + pad,
                    fill=color, outline=color, tags="bbox"
                )
                self.canvas.tag_raise(text_id)

    # ----------------------------------------------------------
    # マウスイベント
    # ----------------------------------------------------------
    def _on_mouse_down(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
        self.temp_rect = None

    def _on_mouse_move(self, event):
        if not self.drawing:
            return
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
        color = CLASS_COLORS.get(self.current_class, "#ffffff")
        self.temp_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline=color, width=2, dash=(4, 4)
        )

    def _on_mouse_up(self, event):
        if not self.drawing:
            return
        self.drawing = False
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y

        # 最小サイズチェック (誤クリック防止)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        # 正規化 (左上が小さい値になるように)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        self.boxes.append((self.current_class, x1, y1, x2, y2))
        self._redraw_boxes()
        self._update_status()

    def _undo_event(self, event):
        self._undo()

    # ----------------------------------------------------------
    # クラス選択
    # ----------------------------------------------------------
    def _select_class(self, class_id):
        self.current_class = class_id
        self._highlight_selected_class()
        self._update_status()

    def _highlight_selected_class(self):
        for cid, btn in self.class_buttons.items():
            color = CLASS_COLORS[cid]
            if cid == self.current_class:
                # 選択中: 白い枠線 + 少し明るい色
                btn.config(relief=tk.RIDGE, bd=3, bg=CLASS_HIGHLIGHT[cid],
                           highlightbackground="white")
            else:
                btn.config(relief=tk.FLAT, bd=1, bg=color,
                           highlightbackground=color)

    # ----------------------------------------------------------
    # アクション
    # ----------------------------------------------------------
    def _auto_detect(self):
        """学習済みモデルを使用して自動アノテーションを行う"""
        if not self.pil_image:
            return

        model_path = os.path.abspath("runs/detect/coin_detect/weights/best.pt")
        if not os.path.exists(model_path):
            messagebox.showwarning("エラー", f"学習済みモデルが見つかりません。\n先に学習を完了してください。\n\nパス: {model_path}")
            return
            
        # 初回のみモデルを遅延ロード (起動を遅くしないため)
        if not hasattr(self, 'yolo_model'):
            try:
                from ultralytics import YOLO
                self.status_bar.config(text="🤖 モデルを読み込んでいます...")
                self.root.update()
                self.yolo_model = YOLO(model_path)
            except Exception as e:
                messagebox.showerror("エラー", f"YOLOモデルの読み込みに失敗しました:\n{e}")
                self._update_status()
                return

        self.status_bar.config(text="✨ 自動検出を実行中...")
        self.root.update()

        try:
            current_conf = self.conf_var.get()
            results = self.yolo_model(self.pil_image, conf=current_conf, verbose=False)
            
            new_boxes_count = 0
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    # 元画像のピクセル座標
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # キャンバス座標にスケーリング
                    cx1 = x1 * self.scale
                    cy1 = y1 * self.scale
                    cx2 = x2 * self.scale
                    cy2 = y2 * self.scale
                    
                    self.boxes.append((cls_id, cx1, cy1, cx2, cy2))
                    new_boxes_count += 1

            self._redraw_boxes()
            self._update_status(f"✨ 自動検出完了: {new_boxes_count} 個の硬貨を追加しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"推論中にエラーが発生しました:\n{e}")
            self._update_status()

    def _prev_image(self):
        if self.current_index > 0:
            self._load_image(self.current_index - 1)

    def _next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self._load_image(self.current_index + 1)

    def _undo(self):
        if self.boxes:
            self.boxes.pop()
            self._redraw_boxes()
            self._update_status()

    def _clear_boxes(self):
        if self.boxes:
            if messagebox.askyesno("確認", "この画像のバウンディングボックスを全て削除しますか？"):
                self.boxes.clear()
                self._redraw_boxes()
                self._update_status()

    def _save_annotations(self, quiet=False):
        """現在の画像のアノテーションをYOLO形式で保存"""
        fname = self.image_files[self.current_index]
        base = os.path.splitext(fname)[0]
        ann_path = os.path.join(self.output_dir, base + ".txt")

        display_w = self.img_width * self.scale
        display_h = self.img_height * self.scale

        lines = []
        for cid, x1, y1, x2, y2 in self.boxes:
            # キャンバス座標 → YOLO正規化座標
            cx = ((x1 + x2) / 2) / display_w
            cy = ((y1 + y2) / 2) / display_h
            w = abs(x2 - x1) / display_w
            h = abs(y2 - y1) / display_h

            # 0〜1 にクリップ
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(ann_path, 'w') as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        if not quiet:
            self._update_status(f"✅ 保存完了: {ann_path} ({len(lines)}個のBBox)")

    def _save_all(self):
        """全画像のアノテーションを保存（現在のものだけ保存）"""
        self._save_annotations(quiet=True)

        # classes.txt も生成
        classes_path = os.path.join(self.output_dir, "classes.txt")
        with open(classes_path, 'w') as f:
            for cid in sorted(CLASSES.keys()):
                f.write(f"{CLASSES[cid]}\n")

        # data.yaml も生成 (YOLOv8学習用)
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {os.path.abspath(self.output_dir)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"nc: {len(CLASSES)}\n")
            f.write(f"names: [{', '.join(repr(CLASSES[i]) for i in sorted(CLASSES.keys()))}]\n")

        messagebox.showinfo(
            "保存完了",
            f"アノテーションを保存しました。\n\n"
            f"・アノテーション: {self.output_dir}/*.txt\n"
            f"・クラス定義: {classes_path}\n"
            f"・データ設定: {yaml_path}"
        )

    def _update_status(self, msg=None):
        if msg:
            self.status_bar.config(text=msg)
        else:
            n = len(self.boxes)
            cls_name = f"{CLASSES.get(self.current_class, '?')}円"
            annotated = sum(
                1 for f in self.image_files
                if os.path.exists(os.path.join(
                    self.output_dir,
                    os.path.splitext(f)[0] + ".txt"
                ))
            )
            self.status_bar.config(
                text=f"選択クラス: {cls_name}  │  "
                     f"BBox数: {n}  │  "
                     f"画像: {self.current_index + 1}/{len(self.image_files)}  │  "
                     f"アノテーション済: {annotated}/{len(self.image_files)}"
            )


# ============================================================
# メイン
# ============================================================

def main():
    # プロジェクトのデフォルトフォルダパス
    default_img_dir = "detection/data/raw"
    default_out_dir = "detection/data/annotations"

    # コマンドライン引数があればそれを優先
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "annotations"
    # 引数がなく、デフォルトフォルダが存在する場合は自動でそれを使用
    elif os.path.exists(default_img_dir):
        image_dir = default_img_dir
        output_dir = default_out_dir
        print(f"📂 デフォルトのフォルダを使用します:")
        print(f"  画像: {image_dir}")
        print(f"  保存先: {output_dir}")
    else:
        # どちらにも該当しない場合はダイアログを表示
        tmp_root = tk.Tk()
        tmp_root.withdraw()
        
        from tkinter import messagebox
        messagebox.showinfo("フォルダ選択", "アノテーションする画像のフォルダ\n（例: detection/data/raw）\nを選択してください。")

        image_dir = filedialog.askdirectory(
            title="アノテーションする画像フォルダを選択",
            initialdir=os.getcwd()
        )
        tmp_root.destroy()

        if not image_dir:
            print("フォルダが選択されませんでした。終了します。")
            sys.exit(0)

        # 出力先: 選択フォルダの親ディレクトリに annotations を作成
        parent = os.path.dirname(image_dir)
        output_dir = os.path.join(parent, "annotations")
        output_dir = os.path.join(parent, "annotations")

    if not os.path.isdir(image_dir):
        print(f"エラー: 画像ディレクトリが見つかりません: {image_dir}")
        print(f"使い方: python annotate.py [画像ディレクトリ] [出力ディレクトリ]")
        sys.exit(1)

    root = tk.Tk()
    root.geometry("1200x850")
    root.minsize(800, 600)

    app = AnnotationTool(root, image_dir, output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
