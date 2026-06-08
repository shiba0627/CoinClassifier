import cv2
import numpy as np
import os
from ultralytics import YOLO

# クラス定義 (detection/annotate.py に合わせる)
CLS_ID_TO_AMOUNT = {
    0: "1",
    1: "5",
    2: "10",
    3: "50",
    4: "100",
    5: "500",
}
AMOUNT_TO_CLS_ID = {v: k for k, v in CLS_ID_TO_AMOUNT.items()}

# BGR colors for each class (matching the annotation tool)
CLASS_COLORS_BGR = {
    0: (87, 71, 255),   # 1Yen
    1: (2, 165, 255),   # 5Yen
    2: (115, 213, 46),  # 10Yen
    3: (255, 144, 30),  # 50Yen
    4: (247, 85, 168),  # 100Yen
    5: (160, 214, 6)    # 500Yen
}

# --- 評価用パラメータ（変更しやすいように上部に配置） ---
IOU_THRESHOLD = 0.5               # 正解と推論のIoU閾値
CLASSIFICATION_CONF_THRESH = 0.4  # Classification時の確信度閾値
DETECTION_CONF_THRESH = 0.3       # Detection時の確信度閾値
# --------------------------------------------------------

def calculate_iou(box1, box2):
    """
    IoU (Intersection over Union) を計算する
    box: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

def read_annotations(txt_path, img_width, img_height):
    """
    YOLO形式のアノテーションを読み込み、ピクセル座標の (cls_id, x1, y1, x2, y2) のリストを返す
    """
    gt_boxes = []
    if not os.path.exists(txt_path):
        return gt_boxes

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx = float(parts[1]) * img_width
            cy = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            gt_boxes.append((cls_id, x1, y1, x2, y2))
            
    return gt_boxes

def predict_classification(img_orig, model):
    """
    Hough変換 + 分類モデルでの推論 (Classification方式)
    """
    pred_boxes = []
    orig_height, orig_width = img_orig.shape[:2]
    img = img_orig.copy()
    
    # リサイズ (処理速度向上とノイズ削減)
    max_size = 1024
    scale = 1.0
    if max(orig_height, orig_width) > max_size:
        scale = max_size / max(orig_height, orig_width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7) 
    
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
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            
            pad = int(r * 0.15)
            r_padded = r + pad
            
            x1 = max(0, int(x - r_padded))
            y1 = max(0, int(y - r_padded))
            x2 = min(img.shape[1], int(x + r_padded))
            y2 = min(img.shape[0], int(y + r_padded))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            coin_crop = img[y1:y2, x1:x2]
            if coin_crop.shape[0] == 0 or coin_crop.shape[1] == 0:
                continue
            
            results = model(coin_crop, verbose=False)
            
            for res in results:
                top1_class = res.names[res.probs.top1]
                conf = res.probs.top1conf.item()
                
                # Classificationの閾値を定数から使用
                if conf > CLASSIFICATION_CONF_THRESH:
                    try:
                        orig_x1 = x1 / scale
                        orig_y1 = y1 / scale
                        orig_x2 = x2 / scale
                        orig_y2 = y2 / scale
                        
                        cls_id = AMOUNT_TO_CLS_ID[str(top1_class)]
                        pred_boxes.append((cls_id, orig_x1, orig_y1, orig_x2, orig_y2, conf))
                    except (ValueError, KeyError):
                        pass
                        
    return pred_boxes

def predict_detection(img_orig, model, conf_thresh=DETECTION_CONF_THRESH):
    """
    YOLO物体検出モデルでの推論 (Detection方式)
    """
    pred_boxes = []
    results = model(img_orig, conf=conf_thresh, verbose=False)
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            pred_boxes.append((cls_id, x1, y1, x2, y2, conf))
            
    return pred_boxes

def evaluate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    TP, FP, FN を計算して返す
    """
    matched_gt = [False] * len(gt_boxes)
    matched_pred = [False] * len(pred_boxes)
    
    # 1. IoUに基づいてマッチング
    iou_matrix = []
    for p_idx, pred in enumerate(pred_boxes):
        for g_idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred[1:5], gt[1:5])
            if iou >= iou_threshold:
                iou_matrix.append((iou, p_idx, g_idx))
                
    iou_matrix.sort(key=lambda x: x[0], reverse=True)
    
    tp = fp = fn = 0
    metrics_per_class = {cid: {'TP': 0, 'FP': 0, 'FN': 0} for cid in CLS_ID_TO_AMOUNT.keys()}
    
    # マッチングしたTPを判定
    for iou, p_idx, g_idx in iou_matrix:
        if not matched_pred[p_idx] and not matched_gt[g_idx]:
            pred_cls = pred_boxes[p_idx][0]
            gt_cls = gt_boxes[g_idx][0]
            
            if pred_cls == gt_cls:
                metrics_per_class[pred_cls]['TP'] += 1
                tp += 1
            else:
                metrics_per_class[pred_cls]['FP'] += 1
                metrics_per_class[gt_cls]['FN'] += 1
                fp += 1
                fn += 1
                
            matched_pred[p_idx] = True
            matched_gt[g_idx] = True
            
    # 2. マッチしなかった推論結果 (FP)
    for p_idx, is_matched in enumerate(matched_pred):
        if not is_matched:
            pred_cls = pred_boxes[p_idx][0]
            metrics_per_class[pred_cls]['FP'] += 1
            fp += 1
            
    # 3. マッチしなかった正解データ (FN)
    for g_idx, is_matched in enumerate(matched_gt):
        if not is_matched:
            gt_cls = gt_boxes[g_idx][0]
            metrics_per_class[gt_cls]['FN'] += 1
            fn += 1
            
    return tp, fp, fn, metrics_per_class

def draw_and_save(img, pred_boxes, tp, fp, fn, out_path, title_text):
    """
    推論結果のバウンディングボックスと評価指標を大きく画像に描画して保存
    """
    # 線幅やフォントのサイズを拡大
    box_thickness = 10
    font_scale = 3.5
    font_thickness = 7
    metrics_font_scale = 4.0
    metrics_font_thickness = 8
    
    img_draw = img.copy()
    
    for p_idx, pred in enumerate(pred_boxes):
        cls_id = pred[0]
        x1, y1, x2, y2 = map(int, pred[1:5])
        conf = pred[5]
        amount_str = f"{CLS_ID_TO_AMOUNT[cls_id]}Yen"
        color = CLASS_COLORS_BGR.get(cls_id, (0, 255, 0))
        
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, box_thickness)
        
        label = f"{amount_str} ({conf:.2f})"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(img_draw, (x1, max(y1 - text_h - 20, 0)), (x1 + text_w, max(y1, text_h + 20)), color, -1)
        cv2.putText(img_draw, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics_text = [
        title_text,
        f"TP: {tp}   FP: {fp}   FN: {fn}",
        f"Precision: {precision:.3f}",
        f"Recall: {recall:.3f}",
        f"F1-Score: {f1:.3f}"
    ]
    
    y_offset = 80
    for text in metrics_text:
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, metrics_font_scale, metrics_font_thickness)
        cv2.rectangle(img_draw, (40, y_offset - text_h - 20), (40 + text_w + 20, y_offset + baseline + 20), (0, 0, 0), -1)
        cv2.putText(img_draw, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, metrics_font_scale, (255, 255, 255), metrics_font_thickness)
        y_offset += text_h + 60
        
    cv2.imwrite(out_path, img_draw)

def summarize_metrics(all_metrics):
    total_tp = sum(m['TP'] for m in all_metrics.values())
    total_fp = sum(m['FP'] for m in all_metrics.values())
    total_fn = sum(m['FN'] for m in all_metrics.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, total_tp, total_fp, total_fn

def evaluate():
    cls_model_path = 'runs/classify/train/weights/best.pt'
    det_model_path = 'runs/detect/coin_detect/weights/best.pt'
    
    print(f"Classificationモデルをロード中: {cls_model_path}")
    cls_model = YOLO(cls_model_path)
    
    print(f"Detectionモデルをロード中: {det_model_path}")
    det_model = YOLO(det_model_path)
    
    test_dir = 'images/test'
    ann_dir = 'images/test/annotations'
    
    out_dir_cls = 'images/results/evaluation_classification'
    out_dir_det = 'images/results/evaluation_detection'
    os.makedirs(out_dir_cls, exist_ok=True)
    os.makedirs(out_dir_det, exist_ok=True)
    
    all_metrics_cls = {cid: {'TP': 0, 'FP': 0, 'FN': 0} for cid in CLS_ID_TO_AMOUNT.keys()}
    all_metrics_det = {cid: {'TP': 0, 'FP': 0, 'FN': 0} for cid in CLS_ID_TO_AMOUNT.keys()}
    
    for i in range(1, 11):
        base_name = f"kadai_{i:02d}"
        
        img_path = os.path.join(test_dir, f"{base_name}.JPG")
        if not os.path.exists(img_path):
            img_path = os.path.join(test_dir, f"{base_name}.jpg")
            
        if not os.path.exists(img_path):
            print(f"スキップ: 画像が見つかりません -> {img_path}")
            continue
            
        ann_path = os.path.join(ann_dir, f"{base_name}.txt")
        if not os.path.exists(ann_path):
            print(f"スキップ: アノテーションファイルが見つかりません -> {ann_path}")
            continue
            
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            continue
            
        orig_height, orig_width = img_orig.shape[:2]
        gt_boxes = read_annotations(ann_path, orig_width, orig_height)
        
        # Classification推論と評価
        pred_cls = predict_classification(img_orig, cls_model)
        tp_cls, fp_cls, fn_cls, metrics_c = evaluate_metrics(pred_cls, gt_boxes, iou_threshold=IOU_THRESHOLD)
        for cid in all_metrics_cls:
            all_metrics_cls[cid]['TP'] += metrics_c[cid]['TP']
            all_metrics_cls[cid]['FP'] += metrics_c[cid]['FP']
            all_metrics_cls[cid]['FN'] += metrics_c[cid]['FN']
            
        out_path_cls = os.path.join(out_dir_cls, f"result_cls_{base_name}.jpg")
        draw_and_save(img_orig, pred_cls, tp_cls, fp_cls, fn_cls, out_path_cls, "Mode: Classification")
        
        # Detection推論と評価
        pred_det = predict_detection(img_orig, det_model, conf_thresh=DETECTION_CONF_THRESH)
        tp_det, fp_det, fn_det, metrics_d = evaluate_metrics(pred_det, gt_boxes, iou_threshold=IOU_THRESHOLD)
        for cid in all_metrics_det:
            all_metrics_det[cid]['TP'] += metrics_d[cid]['TP']
            all_metrics_det[cid]['FP'] += metrics_d[cid]['FP']
            all_metrics_det[cid]['FN'] += metrics_d[cid]['FN']
            
        out_path_det = os.path.join(out_dir_det, f"result_det_{base_name}.jpg")
        draw_and_save(img_orig, pred_det, tp_det, fp_det, fn_det, out_path_det, "Mode: Detection")
        
        print(f"{base_name} 完了: GT={len(gt_boxes)} | Cls予測={len(pred_cls)} | Det予測={len(pred_det)}")
        
    # --- サマリー出力 ---
    p_cls, r_cls, f1_cls, tp_c, fp_c, fn_c = summarize_metrics(all_metrics_cls)
    p_det, r_det, f1_det, tp_d, fp_d, fn_d = summarize_metrics(all_metrics_det)
    
    print("\n" + "="*60)
    print("モデル比較結果サマリー (全画像トータル)")
    print("="*60)
    print(f"{'Mode':<18} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 75)
    print(f"{'Classification':<18} | {tp_c:<4} | {fp_c:<4} | {fn_c:<4} | {p_cls:<9.3f} | {r_cls:<9.3f} | {f1_cls:<9.3f}")
    print(f"{'Detection':<18} | {tp_d:<4} | {fp_d:<4} | {fn_d:<4} | {p_det:<9.3f} | {r_det:<9.3f} | {f1_det:<9.3f}")
    print("="*60)
    print(f"\n[Classification 画像] -> {out_dir_cls}")
    print(f"[Detection 画像]      -> {out_dir_det}")

if __name__ == "__main__":
    evaluate()
