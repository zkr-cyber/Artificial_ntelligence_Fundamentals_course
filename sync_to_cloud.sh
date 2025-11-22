#!/bin/bash
# 同步文件到云服务器的脚本

# 云服务器配置
SSH_HOST="connect.bjb2.seetacloud.com"
SSH_PORT="34066"
SSH_USER="root"
REMOTE_DIR="/root/ma_seg_project"

echo "======================================================================"
echo "同步文件到云服务器"
echo "======================================================================"
echo "目标: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "远程目录: $REMOTE_DIR"
echo "======================================================================"

# 创建远程目录
echo ""
echo "1. 创建远程目录..."
ssh -p $SSH_PORT $SSH_USER@$SSH_HOST "mkdir -p $REMOTE_DIR/scripts $REMOTE_DIR/configs"

# 上传训练脚本
echo ""
echo "2. 上传训练脚本..."
scp -P $SSH_PORT scripts/*.py $SSH_USER@$SSH_HOST:$REMOTE_DIR/scripts/

# 上传配置文件
echo ""
echo "3. 上传配置文件..."
scp -P $SSH_PORT configs/*.yaml $SSH_USER@$SSH_HOST:$REMOTE_DIR/configs/

# 上传主要文件
echo ""
echo "4. 上传主要文件..."
scp -P $SSH_PORT quick_train.py $SSH_USER@$SSH_HOST:$REMOTE_DIR/
scp -P $SSH_PORT cloud_train.py $SSH_USER@$SSH_HOST:$REMOTE_DIR/
scp -P $SSH_PORT requirements.txt $SSH_USER@$SSH_HOST:$REMOTE_DIR/
scp -P $SSH_PORT README_TRAINING.md $SSH_USER@$SSH_HOST:$REMOTE_DIR/

# 上传模型文件
echo ""
echo "5. 上传预训练模型..."
if [ -f "yolo11n-seg.pt" ]; then
    scp -P $SSH_PORT yolo11n-seg.pt $SSH_USER@$SSH_HOST:$REMOTE_DIR/
    echo "✓ 模型文件已上传"
else
    echo "⚠ 未找到 yolo11n-seg.pt，将在云端自动下载"
fi

# 上传数据集（可选，如果需要）
echo ""
echo "6. 数据集同步"
read -p "是否同步数据集到云服务器？这可能需要较长时间 [y/N]: " sync_dataset
if [[ $sync_dataset =~ ^[Yy]$ ]]; then
    echo "同步数据集中..."
    rsync -avz --progress -e "ssh -p $SSH_PORT" \
        --exclude='*.cache' \
        dataset/ $SSH_USER@$SSH_HOST:$REMOTE_DIR/dataset/
    echo "✓ 数据集已同步"
else
    echo "⊘ 跳过数据集同步"
fi

echo ""
echo "======================================================================"
echo "✓ 同步完成！"
echo "======================================================================"
echo ""
echo "下一步:"
echo "1. SSH 连接到云服务器:"
echo "   ssh -p $SSH_PORT $SSH_USER@$SSH_HOST"
echo ""
echo "2. 进入项目目录:"
echo "   cd $REMOTE_DIR"
echo ""
echo "3. 安装依赖（如果需要）:"
echo "   pip install -r requirements.txt"
echo ""
echo "4. 开始训练:"
echo "   # 快速测试（10轮）"
echo "   python cloud_train.py --mode test"
echo ""
echo "   # 标准训练（200轮）"
echo "   python cloud_train.py --mode standard"
echo ""
echo "   # 完整训练（300轮，RTX 5090优化）"
echo "   python cloud_train.py --mode full"
echo ""
echo "   # 医学图像优化"
echo "   python cloud_train.py --mode medical"
echo ""
echo "5. 后台运行（推荐）:"
echo "   nohup python cloud_train.py --mode full > training.log 2>&1 &"
echo "   # 查看日志: tail -f training.log"
echo "======================================================================"

