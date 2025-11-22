"""
一键运行完整实验：训练 + 评估 + 与论文对比
"""
import argparse
import subprocess
import sys
from pathlib import Path
import json


def run_command(cmd: list, description: str):
    """运行命令并处理错误"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n错误: {description}失败")
        sys.exit(1)
    
    print(f"\n{description}完成！")
    return result


def main():
    parser = argparse.ArgumentParser(description="一键运行训练和评估实验")
    
    # 数据集配置
    parser.add_argument("--data", type=str,
                       default=str(Path.cwd() / "configs" / "ma_seg_all.yaml"),
                       help="数据集配置文件")
    parser.add_argument("--eval_root", type=str,
                       default=str(Path.cwd() / "dataset"),
                       help="评估数据集根目录")
    parser.add_argument("--eval_res", type=str, default="1536x",
                       help="评估分辨率")
    parser.add_argument("--eval_subset", type=str, default="i",
                       help="评估子集")
    
    # 模型配置
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt",
                       help="预训练模型")
    parser.add_argument("--project", type=str, default="runs/experiment",
                       help="实验保存路径")
    parser.add_argument("--name", type=str, default=None,
                       help="实验名称")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=300,
                       help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图像大小")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备")
    parser.add_argument("--patience", type=int, default=50,
                       help="早停耐心值")
    
    # 论文对比
    parser.add_argument("--paper_iou", type=float, default=None,
                       help="论文IoU")
    parser.add_argument("--paper_dice", type=float, default=None,
                       help="论文Dice")
    parser.add_argument("--paper_precision", type=float, default=None,
                       help="论文Precision")
    parser.add_argument("--paper_recall", type=float, default=None,
                       help="论文Recall")
    parser.add_argument("--paper_f1", type=float, default=None,
                       help="论文F1")
    
    # 流程控制
    parser.add_argument("--skip_train", action="store_true",
                       help="跳过训练，只进行评估")
    parser.add_argument("--skip_eval", action="store_true",
                       help="跳过评估，只进行训练")
    
    args = parser.parse_args()
    
    # 确定实验名称
    if args.name is None:
        from datetime import datetime
        args.name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    
    # 步骤1: 训练
    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "scripts/train_improved.py",
            "--data", args.data,
            "--model", args.model,
            "--project", args.project,
            "--name", args.name,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--imgsz", str(args.imgsz),
            "--device", args.device,
            "--patience", str(args.patience),
        ]
        
        run_command(train_cmd, "训练模型")
    else:
        print("\n跳过训练步骤")
        if not best_model_path.exists():
            print(f"错误: 模型文件不存在 {best_model_path}")
            sys.exit(1)
    
    # 步骤2: 评估
    if not args.skip_eval:
        if not best_model_path.exists():
            print(f"\n错误: 找不到训练好的模型 {best_model_path}")
            sys.exit(1)
        
        eval_cmd = [
            sys.executable,
            "scripts/eval_improved.py",
            "--root", args.eval_root,
            "--res", args.eval_res,
            "--subset", args.eval_subset,
            "--models", str(best_model_path),
            "--model_names", args.name,
            "--imgsz", str(args.imgsz),
            "--device", args.device,
            "--output", str(Path(args.project) / args.name / "evaluation_results.json"),
        ]
        
        # 添加论文结果参数
        if args.paper_iou is not None:
            eval_cmd.extend(["--paper_iou", str(args.paper_iou)])
        if args.paper_dice is not None:
            eval_cmd.extend(["--paper_dice", str(args.paper_dice)])
        if args.paper_precision is not None:
            eval_cmd.extend(["--paper_precision", str(args.paper_precision)])
        if args.paper_recall is not None:
            eval_cmd.extend(["--paper_recall", str(args.paper_recall)])
        if args.paper_f1 is not None:
            eval_cmd.extend(["--paper_f1", str(args.paper_f1)])
        
        run_command(eval_cmd, "评估模型")
    else:
        print("\n跳过评估步骤")
    
    # 完成
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print(f"实验路径: {Path(args.project) / args.name}")
    print(f"最佳模型: {best_model_path}")
    if not args.skip_eval:
        print(f"评估结果: {Path(args.project) / args.name / 'evaluation_results.json'}")
    print("="*80)


if __name__ == "__main__":
    main()

