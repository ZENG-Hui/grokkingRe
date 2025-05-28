#PBS -N test
#PBS -l nodes=1:ppn=24
#PBS -l walltime=1:00:00
#PBS -q cmt
ulimit -s unlimited

cd $PBS_O_WORKDIR
cp $PBS_NODEFILE node
NCORE=`cat node | wc -l`

# 创建调试日志
echo "开始执行时间: $(date)" > debug.log
echo "当前工作目录: $(pwd)" >> debug.log
echo "节点核心数: $NCORE" >> debug.log

# 1. 激活适当的Python环境
source ~/.bashrc
conda activate base
echo "环境激活状态: $?" >> debug.log

# 2. 设置wandb
export WANDB_MODE=offline
export WANDB_API_KEY=2695076dd0f87f2dba081c03fdc9cfa84acef643
export WANDB_PROJECT=grokking
export WANDB_NAME="w003_test"
echo "WANDB设置完成" >> debug.log

# 3. 检查并记录wandb目录初始状态
echo "训练前wandb目录状态:" >> debug.log
if [ -d "wandb" ]; then
    ls -la wandb/ >> debug.log
else
    echo "wandb目录不存在，将创建" >> debug.log
    mkdir -p wandb
fi

# 4. 运行Python脚本并将输出重定向到日志文件
echo "开始运行训练脚本: $(date)" >> debug.log
python /home/zengh/2025/Grokking/grokkingRe/grokking/cli.py \
  > training.log 2>&1
TRAIN_STATUS=$?
echo "训练脚本退出状态: $TRAIN_STATUS (0表示正常结束)" >> debug.log

# 5. 等待确保wandb目录创建完成
echo "等待wandb目录创建完成..." >> debug.log
sleep 10

# 6. 使用更可靠的方式查找wandb目录
echo "训练后wandb目录状态:" >> debug.log
if [ -d "wandb" ]; then
    ls -la wandb/ >> debug.log
    
    # 使用stat命令按修改时间排序目录
    WANDB_DIRS=$(find wandb -type d \( -name "*run*" \) -not -path "*/\.*" | 
                 xargs stat --format="%Y %n" 2>/dev/null | 
                 sort -nr | 
                 cut -d' ' -f2-)
    
    echo "按时间排序的wandb运行目录:" >> debug.log
    echo "$WANDB_DIRS" >> debug.log
    
    # 获取时间最新的运行目录
    WANDB_DIR=$(echo "$WANDB_DIRS" | head -n 1)
    
    if [ -n "$WANDB_DIR" ]; then
        echo "最新的wandb运行目录: $WANDB_DIR" > wandb_latest_run.txt
        echo "已更新wandb_latest_run.txt文件" >> debug.log
        cat wandb_latest_run.txt >> debug.log
    else
        echo "未找到有效的wandb运行目录" > wandb_latest_run.txt
        echo "未找到有效的wandb运行目录" >> debug.log
    fi
else
    echo "训练后wandb目录不存在" >> debug.log
    echo "wandb目录不存在" > wandb_latest_run.txt
fi

# 7. 完成记录
echo "脚本执行完成时间: $(date)" >> debug.log

echo "训练完成，准备同步数据..." >> debug.log
# 通知主节点开始同步
touch need_sync.flag  