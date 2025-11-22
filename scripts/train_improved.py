"""
改进的YOLOv8分割模型训练脚本
支持MPS加速、高级训练配置和实验跟踪
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
import yaml


def check_device():
    """检测可用的设备"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def save_config(args, save_dir: Path):
    """保存训练配置"""
    config = vars(args)
    config_file = save_dir / "training_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"配置已保存到: {config_file}")


def main():
    parser = argparse.ArgumentParser(description="改进的YOLO分割模型训练脚本")
    
    # 基础配置
    parser.add_argument("--data", type=str, 
                       default=str(Path.cwd() / "configs" / "ma_seg_all.yaml"),
                       help="数据集配置文件路径")
    parser.add_argument("--model", type=str, 
                       default="yolo11n-seg.pt",
                       help="预训练模型路径")
    parser.add_argument("--project", type=str, 
                       default="runs/improved_seg",
                       help="项目保存路径")
    parser.add_argument("--name", type=str, 
                       default=None,
                       help="实验名称（默认：时间戳）")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=300,
                       help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="输入图像大小")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备选择: auto/mps/cuda/cpu")
    
    # 优化器配置
    parser.add_argument("--optimizer", type=str, default="AdamW",
                       choices=["SGD", "Adam", "AdamW", "RMSProp"],
                       help="优化器类型")
    parser.add_argument("--lr0", type=float, default=0.001,
                       help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01,
                       help="最终学习率（相对于初始学习率）")
    parser.add_argument("--momentum", type=float, default=0.937,
                       help="SGD动量/Adam beta1")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                       help="权重衰减")
    parser.add_argument("--warmup_epochs", type=float, default=3.0,
                       help="学习率预热轮数")
    parser.add_argument("--warmup_momentum", type=float, default=0.8,
                       help="预热初始动量")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1,
                       help="预热bias学习率")
    
    # 数据增强
    parser.add_argument("--hsv_h", type=float, default=0.015,
                       help="图像HSV色调增强")
    parser.add_argument("--hsv_s", type=float, default=0.7,
                       help="图像HSV饱和度增强")
    parser.add_argument("--hsv_v", type=float, default=0.4,
                       help="图像HSV明度增强")
    parser.add_argument("--degrees", type=float, default=0.0,
                       help="图像旋转角度")
    parser.add_argument("--translate", type=float, default=0.1,
                       help="图像平移")
    parser.add_argument("--scale", type=float, default=0.5,
                       help="图像缩放")
    parser.add_argument("--shear", type=float, default=0.0,
                       help="图像剪切")
    parser.add_argument("--perspective", type=float, default=0.0,
                       help="图像透视变换")
    parser.add_argument("--flipud", type=float, default=0.0,
                       help="上下翻转概率")
    parser.add_argument("--fliplr", type=float, default=0.5,
                       help="左右翻转概率")
    parser.add_argument("--mosaic", type=float, default=1.0,
                       help="马赛克增强概率")
    parser.add_argument("--mixup", type=float, default=0.0,
                       help="Mixup增强概率")
    parser.add_argument("--copy_paste", type=float, default=0.0,
                       help="Copy-paste增强概率")
    
    # 高级配置
    parser.add_argument("--patience", type=int, default=50,
                       help="早停耐心值（轮数）")
    parser.add_argument("--save_period", type=int, default=-1,
                       help="每N个epoch保存一次检查点（-1表示只在最后保存）")
    parser.add_argument("--cache", type=str, default="False",
                       choices=["True", "False", "disk", "ram"],
                       help="图像缓存策略")
    parser.add_argument("--workers", type=int, default=8,
                       help="数据加载工作线程数")
    parser.add_argument("--close_mosaic", type=int, default=10,
                       help="在最后N个epoch关闭mosaic增强")
    
    # 损失函数权重
    parser.add_argument("--box", type=float, default=7.5,
                       help="边界框损失权重")
    parser.add_argument("--cls", type=float, default=0.5,
                       help="分类损失权重")
    parser.add_argument("--dfl", type=float, default=1.5,
                       help="DFL损失权重")
    
    # 验证和可视化
    parser.add_argument("--val", type=bool, default=True,
                       help="训练期间是否验证")
    parser.add_argument("--plots", type=bool, default=True,
                       help="是否保存训练图表")
    parser.add_argument("--verbose", type=bool, default=True,
                       help="是否打印详细信息")
    
    # 其他选项
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--resume", type=str, default=None,
                       help="从检查点恢复训练")
    parser.add_argument("--amp", type=bool, default=True,
                       help="是否使用自动混合精度训练")
    parser.add_argument("--fraction", type=float, default=1.0,
                       help="使用数据集的比例（用于测试）")
    
    args = parser.parse_args()
    
    # 设备检测和配置
    if args.device == "auto":
        detected_device = check_device()
        print(f"自动检测设备: {detected_device}")
        args.device = detected_device
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("警告: MPS不可用，切换到CPU")
        args.device = "cpu"
    
    print(f"使用设备: {args.device}")
    
    # 设置实验名称
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"train_{timestamp}"
    
    # 创建保存目录
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    save_config(args, save_dir)
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 训练配置
    train_args = {
        # 基础配置
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        
        # 优化器
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        
        # 数据增强
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        
        # 高级配置
        "patience": args.patience,
        "save_period": args.save_period,
        "cache": args.cache,
        "workers": args.workers,
        "close_mosaic": args.close_mosaic,
        
        # 损失函数
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        
        # 验证和可视化
        "val": args.val,
        "plots": args.plots,
        "verbose": args.verbose,
        
        # 其他
        "seed": args.seed,
        "amp": args.amp,
        "fraction": args.fraction,
    }
    
    # 恢复训练
    if args.resume:
        train_args["resume"] = args.resume
        print(f"从检查点恢复训练: {args.resume}")
    
    # 打印训练配置摘要
    print("\n" + "="*60)
    print("训练配置摘要")
    print("="*60)
    print(f"数据集: {args.data}")
    print(f"模型: {args.model}")
    print(f"设备: {args.device}")
    print(f"轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"图像大小: {args.imgsz}")
    print(f"优化器: {args.optimizer}")
    print(f"初始学习率: {args.lr0}")
    print(f"最终学习率: {args.lr0 * args.lrf}")
    print(f"早停耐心值: {args.patience}")
    print(f"保存路径: {save_dir}")
    print("="*60 + "\n")
    
    # 开始训练
    print("开始训练...\n")
    results = model.train(**train_args)
    
    # 训练完成
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"模型保存在: {save_dir}")
    print(f"最佳模型: {save_dir / 'weights' / 'best.pt'}")
    print(f"最后模型: {save_dir / 'weights' / 'last.pt'}")
    
    # 保存训练结果摘要
    if hasattr(results, 'results_dict'):
        results_file = save_dir / "training_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results.results_dict, f, indent=2, ensure_ascii=False)
        print(f"训练结果保存在: {results_file}")
    
    print("\n现在可以运行评估脚本与论文结果比较：")
    print(f"python scripts/eval_compare_paper.py --yolov8 {save_dir / 'weights' / 'best.pt'}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()

