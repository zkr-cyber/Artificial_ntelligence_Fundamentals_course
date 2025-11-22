#!/usr/bin/env python3
"""
云服务器训练脚本 - 针对 RTX 5090 优化
适配 PyTorch 2.8.0 + CUDA 12.8
"""

import sys
import subprocess
from pathlib import Path
import torch


def check_cloud_environment():
    """检查云服务器环境"""
    print("="*60)
    print("云服务器环境检查")
    print("="*60)
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.version.cuda}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("✗ CUDA不可用")
        sys.exit(1)
    
    # 检查ultralytics
    try:
        import ultralytics
        print(f"✓ Ultralytics版本: {ultralytics.__version__}")
    except ImportError:
        print("✗ 未安装ultralytics")
        print("  安装中...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"])
    
    print("="*60 + "\n")
    
    return torch.cuda.device_count()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="云服务器训练启动器（RTX 5090优化）")
    
    # 训练模式
    parser.add_argument("--mode", type=str, default="full",
                       choices=["test", "standard", "full", "medical"],
                       help="训练模式")
    parser.add_argument("--data", type=str, 
                       default="configs/ma_seg_all.yaml",
                       help="数据集配置文件")
    parser.add_argument("--name", type=str, default=None,
                       help="实验名称")
    
    # RTX 5090 优化参数
    parser.add_argument("--batch", type=int, default=None,
                       help="批次大小（默认根据模式自动设置）")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图像尺寸")
    parser.add_argument("--workers", type=int, default=16,
                       help="数据加载线程数（RTX 5090建议16+）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=None,
                       help="训练轮数（默认根据模式自动设置）")
    parser.add_argument("--patience", type=int, default=None,
                       help="早停耐心值（默认根据模式自动设置）")
    
    # 高级选项
    parser.add_argument("--amp", action="store_true", default=True,
                       help="启用自动混合精度（AMP）训练")
    parser.add_argument("--cache", type=str, default="ram",
                       choices=["False", "disk", "ram"],
                       help="缓存策略（RTX 5090建议ram）")
    parser.add_argument("--resume", type=str, default=None,
                       help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    # 环境检查
    gpu_count = check_cloud_environment()
    
    # 根据模式配置参数
    if args.mode == "test":
        print("📝 模式: 快速测试（验证环境）")
        epochs = args.epochs or 10
        batch = args.batch or 32
        patience = args.patience or 10
        name = args.name or "cloud_test"
        optimizer_config = {}
        
    elif args.mode == "standard":
        print("📝 模式: 标准训练")
        epochs = args.epochs or 200
        batch = args.batch or 32
        patience = args.patience or 50
        name = args.name or "cloud_standard"
        optimizer_config = {
            "--optimizer": "AdamW",
            "--lr0": "0.001",
            "--weight_decay": "0.0005",
        }
        
    elif args.mode == "full":
        print("📝 模式: 完整训练（RTX 5090优化）")
        epochs = args.epochs or 300
        batch = args.batch or 40  # RTX 5090 可以处理更大批次
        patience = args.patience or 80
        name = args.name or "cloud_full_rtx5090"
        optimizer_config = {
            "--optimizer": "AdamW",
            "--lr0": "0.001",
            "--lrf": "0.01",
            "--weight_decay": "0.001",
            "--warmup_epochs": "10",
        }
        
    elif args.mode == "medical":
        print("📝 模式: 医学图像优化（RTX 5090 + 高级增强）")
        epochs = args.epochs or 300
        batch = args.batch or 32
        patience = args.patience or 100
        name = args.name or "cloud_medical_rtx5090"
        optimizer_config = {
            "--optimizer": "AdamW",
            "--lr0": "0.0005",
            "--lrf": "0.01",
            "--weight_decay": "0.001",
            "--warmup_epochs": "15",
            "--hsv_h": "0.01",
            "--hsv_s": "0.5",
            "--hsv_v": "0.3",
            "--degrees": "5",
            "--mosaic": "0.8",
            "--flipud": "0.5",
            "--close_mosaic": "30",
            "--mixup": "0.1",
        }
    
    # 构建训练命令
    cmd = [
        sys.executable,
        "scripts/train_improved.py",
        "--data", args.data,
        "--model", "yolo11n-seg.pt",
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--imgsz", str(args.imgsz),
        "--device", "cuda",
        "--patience", str(patience),
        "--name", name,
        "--workers", str(args.workers),
        "--cache", args.cache,
    ]
    
    # 添加 AMP
    if args.amp:
        cmd.extend(["--amp", "True"])
    
    # 添加优化器配置
    for key, value in optimizer_config.items():
        cmd.extend([key, value])
    
    # 恢复训练
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    # 打印配置
    print("="*60)
    print("训练配置（RTX 5090 优化）")
    print("="*60)
    print(f"数据集: {args.data}")
    print(f"GPU: CUDA (RTX 5090)")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch} (充分利用显存)")
    print(f"图像尺寸: {args.imgsz}")
    print(f"工作线程: {args.workers}")
    print(f"缓存策略: {args.cache}")
    print(f"混合精度: {'启用' if args.amp else '禁用'}")
    print(f"早停耐心值: {patience}")
    print(f"实验名称: {name}")
    print("="*60 + "\n")
    
    # 显示命令
    print("执行命令:")
    print(" ".join(cmd))
    print("\n" + "="*60 + "\n")
    
    # 开始训练（自动运行，无需确认）
    print("开始训练...")
    print("="*60 + "\n")
    
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✓ 训练完成！")
            print("="*60)
            
            # 训练结果路径
            best_model = f"runs/improved_seg/{name}/weights/best.pt"
            print(f"\n最佳模型: {best_model}")
            print("\n下一步:")
            print(f"1. 评估模型:")
            print(f"   python scripts/eval_improved.py --models {best_model} --device cuda")
            print(f"\n2. 下载模型到本地:")
            print(f"   scp -P 34066 root@connect.bjb2.seetacloud.com:/root/ma_seg_project/{best_model} .")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("✗ 训练失败")
            print("="*60)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

