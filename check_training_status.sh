#!/bin/bash

# 检查训练状态的快速脚本

TRAIN_DIR="runs/improved_seg/continue_to_300"
RESULTS_CSV="$TRAIN_DIR/results.csv"

echo "========================================"
echo "训练状态检查"
echo "========================================"
echo ""

# 检查训练目录
if [ -d "$TRAIN_DIR" ]; then
    echo "✓ 训练目录存在: $TRAIN_DIR"
else
    echo "✗ 训练目录不存在"
    exit 1
fi

# 检查结果文件
if [ -f "$RESULTS_CSV" ]; then
    echo "✓ 结果文件存在"
    echo ""
    
    # 统计训练轮数
    TOTAL_EPOCHS=$(tail -n +2 "$RESULTS_CSV" | wc -l | tr -d ' ')
    echo "已完成轮数: $TOTAL_EPOCHS"
    
    # 显示最新几轮的结果
    echo ""
    echo "最近的训练结果:"
    echo "----------------------------------------"
    tail -n 5 "$RESULTS_CSV" | column -t -s,
    
else
    echo "⏳ 结果文件尚未生成（训练刚开始）"
    echo ""
    echo "请稍等片刻后再检查"
fi

echo ""
echo "========================================"
echo "实时监控命令:"
echo "python monitor_training.py"
echo ""
echo "查看日志:"
echo "tail -f $RESULTS_CSV"
echo "========================================"

