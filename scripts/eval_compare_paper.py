import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO

def polygons_to_mask(polys, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        xs = np.clip(pts[:, 0], 0, w - 1).astype(np.int32)
        ys = np.clip(pts[:, 1], 0, h - 1).astype(np.int32)
        mask[ys, xs] = 1
    return mask

def load_gt_union_masks(images_dir: Path, coco_json: Path):
    coco = json.loads(coco_json.read_text())
    by_img = {}
    for ann in coco.get("annotations", []):
        by_img.setdefault(ann["image_id"], []).append(ann)
    masks = {}
    for info in coco["images"]:
        w, h = info["width"], info["height"]
        union = np.zeros((h, w), dtype=np.uint8)
        for ann in by_img.get(info["id"], []):
            seg = ann.get("segmentation")
            if isinstance(seg, list):
                union = np.maximum(union, polygons_to_mask(seg, w, h))
        masks[info["file_name"]] = union
    return masks

def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(np.uint8)
    gt = gt_mask.astype(np.uint8)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum() - inter
    iou = inter / (union + 1e-6)
    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
    return iou, dice

def eval_yolo(model_path: str, images_dir: Path, gt_masks: dict, imgsz: int, th: float, device: str = "auto"):
    model = YOLO(model_path)
    ious, dices = [], []
    for name, gt in gt_masks.items():
        img_path = images_dir / name
        res = model.predict(str(img_path), imgsz=imgsz, conf=0.25, verbose=False, device=device)
        union = np.zeros(gt.shape, dtype=np.uint8)
        for r in res:
            if r.masks is not None and hasattr(r.masks, "data"):
                ms = r.masks.data.cpu().numpy()
                for m in ms:
                    m_bin = (m > th).astype(np.uint8)
                    union = np.maximum(union, m_bin)
        iou, dice = compute_metrics(union, gt)
        ious.append(iou)
        dices.append(dice)
    return float(np.mean(ious)), float(np.mean(dices))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.cwd() / "dataset"))
    ap.add_argument("--res", default="1536x")
    ap.add_argument("--subset", default="i")
    ap.add_argument("--yolov8", default="yolo11n-seg.pt")
    ap.add_argument("--yolov9", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--paper_iou", type=float, default=None)
    ap.add_argument("--paper_dice", type=float, default=None)
    args = ap.parse_args()

    path_root = Path(args.root) / args.res / args.subset
    images_val = path_root / "im_val"
    coco_val = path_root / "label" / "val.json"
    gt_masks = load_gt_union_masks(images_val, coco_val)

    v8_iou, v8_dice = eval_yolo(args.yolov8, images_val, gt_masks, args.imgsz, args.th, args.device)
    out = {
        "yolov8": {"iou": v8_iou, "dice": v8_dice},
    }
    if args.yolov9:
        v9_iou, v9_dice = eval_yolo(args.yolov9, images_val, gt_masks, args.imgsz, args.th, args.device)
        out["yolov9"] = {"iou": v9_iou, "dice": v9_dice}

    if args.paper_iou is not None and args.paper_dice is not None:
        out["paper"] = {"iou": args.paper_iou, "dice": args.paper_dice}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()