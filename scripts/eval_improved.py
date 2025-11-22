"""
改进的评估脚本，用于计算更多指标并与论文结果对比
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import cv2


def polygons_to_mask(polys, w, h):
    """将多边形转换为掩码"""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.clip(pts, 0, [w-1, h-1])
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def load_gt_masks(images_dir: Path, coco_json: Path):
    """加载真实标注掩码"""
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
                mask = polygons_to_mask(seg, w, h)
                union = np.maximum(union, mask)
        masks[info["file_name"]] = union
    
    return masks


def compute_metrics(pred_mask, gt_mask):
    """计算评估指标"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    # 交并比 (IoU)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / (union + 1e-8)
    
    # Dice系数
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-8)
    
    # 精确率和召回率
    tp = intersection
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # F1分数
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def eval_model(model_path: str, images_dir: Path, gt_masks: dict, 
               imgsz: int, conf_thresh: float, mask_thresh: float, 
               device: str = "auto"):
    """评估YOLO模型"""
    print(f"\n评估模型: {model_path}")
    print(f"设备: {device}")
    print(f"图像大小: {imgsz}")
    print(f"置信度阈值: {conf_thresh}")
    print(f"掩码阈值: {mask_thresh}")
    
    model = YOLO(model_path)
    
    all_metrics = []
    
    for name, gt in tqdm(gt_masks.items(), desc="评估进度"):
        img_path = images_dir / name
        if not img_path.exists():
            print(f"警告: 图像不存在 {img_path}")
            continue
        
        # 预测
        results = model.predict(
            str(img_path), 
            imgsz=imgsz, 
            conf=conf_thresh, 
            verbose=False, 
            device=device
        )
        
        # 合并所有预测掩码
        pred_mask = np.zeros(gt.shape, dtype=np.uint8)
        for r in results:
            if r.masks is not None and hasattr(r.masks, "data"):
                masks = r.masks.data.cpu().numpy()
                for m in masks:
                    # 调整掩码大小
                    m_resized = cv2.resize(m, (gt.shape[1], gt.shape[0]))
                    m_bin = (m_resized > mask_thresh).astype(np.uint8)
                    pred_mask = np.maximum(pred_mask, m_bin)
        
        # 计算指标
        metrics = compute_metrics(pred_mask, gt)
        all_metrics.append(metrics)
    
    # 汇总统计
    summary = {}
    for key in ["iou", "dice", "precision", "recall", "f1"]:
        values = [m[key] for m in all_metrics]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values))
        }
    
    return summary, all_metrics


def print_comparison_table(results: dict):
    """打印对比表格"""
    print("\n" + "="*80)
    print("评估结果对比")
    print("="*80)
    
    # 表头
    print(f"{'模型':<20} {'IoU':<15} {'Dice':<15} {'Precision':<15} {'Recall':<15} {'F1':<15}")
    print("-"*80)
    
    # 打印每个模型的结果
    for model_name, metrics in results.items():
        if model_name != "paper":
            iou = f"{metrics['iou']['mean']:.4f}±{metrics['iou']['std']:.4f}"
            dice = f"{metrics['dice']['mean']:.4f}±{metrics['dice']['std']:.4f}"
            precision = f"{metrics['precision']['mean']:.4f}±{metrics['precision']['std']:.4f}"
            recall = f"{metrics['recall']['mean']:.4f}±{metrics['recall']['std']:.4f}"
            f1 = f"{metrics['f1']['mean']:.4f}±{metrics['f1']['std']:.4f}"
        else:
            # 论文结果（假设只有均值）
            iou = f"{metrics.get('iou', 0):.4f}"
            dice = f"{metrics.get('dice', 0):.4f}"
            precision = f"{metrics.get('precision', 0):.4f}"
            recall = f"{metrics.get('recall', 0):.4f}"
            f1 = f"{metrics.get('f1', 0):.4f}"
        
        print(f"{model_name:<20} {iou:<15} {dice:<15} {precision:<15} {recall:<15} {f1:<15}")
    
    print("="*80)
    
    # 打印改进情况
    if "paper" in results and len(results) > 1:
        print("\n相对于论文的改进:")
        print("-"*80)
        for model_name, metrics in results.items():
            if model_name != "paper":
                for metric_name in ["iou", "dice", "precision", "recall", "f1"]:
                    if metric_name in results["paper"]:
                        paper_val = results["paper"][metric_name]
                        model_val = metrics[metric_name]["mean"]
                        improvement = ((model_val - paper_val) / paper_val) * 100
                        print(f"{model_name} {metric_name.upper()}: {improvement:+.2f}%")
        print("-"*80)


def main():
    parser = argparse.ArgumentParser(description="改进的YOLO模型评估脚本")
    
    # 数据集配置
    parser.add_argument("--root", type=str, 
                       default=str(Path.cwd() / "dataset"),
                       help="数据集根目录")
    parser.add_argument("--res", type=str, default="1536x",
                       help="分辨率")
    parser.add_argument("--subset", type=str, default="i",
                       help="子集")
    
    # 模型配置
    parser.add_argument("--models", nargs="+", 
                       default=["yolo11n-seg.pt"],
                       help="要评估的模型路径列表")
    parser.add_argument("--model_names", nargs="+",
                       default=None,
                       help="模型名称（用于显示）")
    
    # 评估参数
    parser.add_argument("--imgsz", type=int, default=640,
                       help="输入图像大小")
    parser.add_argument("--conf_thresh", type=float, default=0.25,
                       help="置信度阈值")
    parser.add_argument("--mask_thresh", type=float, default=0.5,
                       help="掩码阈值")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备: auto/mps/cuda/cpu")
    
    # 论文结果（用于对比）
    parser.add_argument("--paper_iou", type=float, default=None,
                       help="论文中的IoU值")
    parser.add_argument("--paper_dice", type=float, default=None,
                       help="论文中的Dice值")
    parser.add_argument("--paper_precision", type=float, default=None,
                       help="论文中的Precision值")
    parser.add_argument("--paper_recall", type=float, default=None,
                       help="论文中的Recall值")
    parser.add_argument("--paper_f1", type=float, default=None,
                       help="论文中的F1值")
    
    # 输出配置
    parser.add_argument("--output", type=str, 
                       default="evaluation_results.json",
                       help="结果保存路径")
    
    args = parser.parse_args()
    
    # 设备检测
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
        print(f"自动检测设备: {args.device}")
    
    # 数据集路径
    path_root = Path(args.root) / args.res / args.subset
    images_val = path_root / "im_val"
    coco_val = path_root / "label" / "val.json"
    
    if not coco_val.exists():
        print(f"错误: 验证集标注文件不存在 {coco_val}")
        return
    
    # 加载真实标注
    print("加载真实标注...")
    gt_masks = load_gt_masks(images_val, coco_val)
    print(f"加载了 {len(gt_masks)} 张图像的标注")
    
    # 评估所有模型
    results = {}
    
    # 设置模型名称
    if args.model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(args.models))]
    else:
        model_names = args.model_names
    
    for model_path, model_name in zip(args.models, model_names):
        if not Path(model_path).exists():
            print(f"警告: 模型文件不存在 {model_path}，跳过")
            continue
        
        summary, details = eval_model(
            model_path, 
            images_val, 
            gt_masks,
            args.imgsz,
            args.conf_thresh,
            args.mask_thresh,
            args.device
        )
        
        results[model_name] = summary
    
    # 添加论文结果
    paper_metrics = {}
    if args.paper_iou is not None:
        paper_metrics["iou"] = args.paper_iou
    if args.paper_dice is not None:
        paper_metrics["dice"] = args.paper_dice
    if args.paper_precision is not None:
        paper_metrics["precision"] = args.paper_precision
    if args.paper_recall is not None:
        paper_metrics["recall"] = args.paper_recall
    if args.paper_f1 is not None:
        paper_metrics["f1"] = args.paper_f1
    
    if paper_metrics:
        results["paper"] = paper_metrics
    
    # 打印对比表格
    print_comparison_table(results)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

