#!/bin/bash

# 配置参数
SYNC_INTERVAL=30  # 每30秒同步一次
SYNC_LOG="monitor/auto_sync.log"
SYNCED_RUNS_FILE="monitor/synced_runs.txt"
COMPUTE_NODE=$(head -n 1 node)  # 从node文件获取计算节点名称
REMOTE_PATH="/home/zengh/2025/Grokking/grokkingRe"  # 计算节点上的项目路径
SKIPPED_RUNS_FILE="monitor/skipped_runs.txt"  # 跳过云端被删除的运行ID列表

SYNC_MODE="all"  # 默认同步所有实验
if [ "$1" = "--latest" ]; then
    SYNC_MODE="latest"
    echo "仅同步最新实验模式已启用" >> $SYNC_LOG
fi

# 添加状态跟踪变量，任务结束后停止同步
TASK_RUNNING=false

# 创建必要的目录
mkdir -p monitor
touch "$SYNCED_RUNS_FILE"
touch "$SKIPPED_RUNS_FILE" 

# 记录脚本启动
echo "启动自动同步 - $(date)" > $SYNC_LOG
echo "配置：同步间隔=$SYNC_INTERVAL秒，计算节点=$COMPUTE_NODE" >> $SYNC_LOG

# 激活Python环境 - 解决wandb命令未找到问题
echo "激活Python环境..." >> $SYNC_LOG
source ~/.bashrc
conda activate base
echo "Python环境激活状态: $?" >> $SYNC_LOG
# 检查wandb是否可用
echo "检查wandb是否可用..." >> $SYNC_LOG
if python -c "import wandb" 2>/dev/null; then
    echo "wandb Python包可用" >> $SYNC_LOG
    # 使用python -m方式调用wandb
    WANDB_CMD="python -m wandb"
else
    echo "错误: wandb Python包不可用，尝试安装..." >> $SYNC_LOG
    pip install wandb >> $SYNC_LOG 2>&1
    if python -c "import wandb" 2>/dev/null; then
        echo "wandb安装成功" >> $SYNC_LOG
        WANDB_CMD="python -m wandb"
    else
        echo "无法安装wandb，同步将失败" >> $SYNC_LOG
        exit 1
    fi
fi
echo "wandb命令设置为: $WANDB_CMD" >> $SYNC_LOG

while true; do
    # 1. 从计算节点获取最新数据
    echo "从计算节点 $COMPUTE_NODE 获取数据 - $(date)" >> $SYNC_LOG
    
    # 检查是否有训练任务
    if ssh $COMPUTE_NODE "[ -f $REMOTE_PATH/need_sync.flag ]"; then
        # 任务正在运行
        TASK_WAS_RUNNING=true
        echo "检测到训练任务，开始同步数据" >> $SYNC_LOG
        
        # 从计算节点同步wandb目录
        rsync -az --exclude="*.tmp" $COMPUTE_NODE:$REMOTE_PATH/wandb/ wandb/ >> $SYNC_LOG 2>&1
        
        # 同步训练日志以便查看训练进度
        rsync -az $COMPUTE_NODE:$REMOTE_PATH/training.log monitor/ >> $SYNC_LOG 2>&1
        rsync -az $COMPUTE_NODE:$REMOTE_PATH/debug.log monitor/ >> $SYNC_LOG 2>&1
        
        echo "数据同步完成 - $(date)" >> $SYNC_LOG
    else
        # 任务未运行
        if [ "$TASK_WAS_RUNNING" = true ]; then
            # 任务状态从运行变为停止，执行最后一次同步
            echo "检测到训练任务已停止，执行最后一次同步..." >> $SYNC_LOG
            
            # 最后一次从计算节点同步数据
            rsync -az --exclude="*.tmp" $COMPUTE_NODE:$REMOTE_PATH/wandb/ wandb/ >> $SYNC_LOG 2>&1
            rsync -az $COMPUTE_NODE:$REMOTE_PATH/training.log monitor/ >> $SYNC_LOG 2>&1
            rsync -az $COMPUTE_NODE:$REMOTE_PATH/debug.log monitor/ >> $SYNC_LOG 2>&1
            
            # 向wandb进行最后一次同步
            echo "开始最后一次wandb同步..." >> $SYNC_LOG
            
            find wandb -type d -name "*run-*" | while read RUN_DIR; do
                echo "最终同步 $RUN_DIR - $(date)" >> $SYNC_LOG
                $WANDB_CMD sync --include-synced "$RUN_DIR" >> $SYNC_LOG 2>&1
            done
            
            echo "最后一次同步完成，监控任务结束 - $(date)" >> $SYNC_LOG
            exit 0
        else
            echo "未检测到运行中的训练任务" >> $SYNC_LOG
        fi
    fi
    
    # 2. 向wandb同步数据
    echo "开始向wandb同步数据 - $(date)" >> $SYNC_LOG

    # 预处理：从现有日志中提取被删除的运行ID
    echo "检查历史日志中的已删除运行..." >> $SYNC_LOG
    grep -E "previously created and deleted" "$SYNC_LOG" | grep -E "run [a-z0-9]+" | \
        awk '{print $5}' | sort -u > /tmp/deleted_run_ids.txt

    # 如果找到被删除的运行ID，添加到跳过列表
    if [ -s /tmp/deleted_run_ids.txt ]; then
        echo "从历史日志中发现以下已删除的运行ID:" >> $SYNC_LOG
        cat /tmp/deleted_run_ids.txt >> $SYNC_LOG
        
        # 查找完整目录路径并添加到跳过列表
        while read -r run_id; do
            # 提取纯ID部分（如果有格式如"run dmbxelxq"）
            id_only=$(echo "$run_id" | grep -oE "[a-z0-9]+")
            
            # 找到包含此ID的目录
            matching_dir=$(find wandb -type d -name "*$id_only*" | head -n 1)
            if [ -n "$matching_dir" ] && ! grep -q "^$matching_dir$" "$SKIPPED_RUNS_FILE"; then
                echo "$matching_dir" >> "$SKIPPED_RUNS_FILE"
                echo "添加到跳过列表: $matching_dir (ID: $id_only)" >> $SYNC_LOG
            fi
        done < /tmp/deleted_run_ids.txt
        
        # 显示当前跳过列表
        echo "当前跳过列表包含以下目录:" >> $SYNC_LOG
        cat "$SKIPPED_RUNS_FILE" >> $SYNC_LOG
    fi

    # 分为两种模式：最新实验和所有实验
    if [ "$SYNC_MODE" = "latest" ]; then
        # 仅同步最新实验
        LATEST_RUN=$(find wandb -type d -name "offline-run-*" | sort -r | head -n 1)
        # 仅同步最新实验的代码
        if [ -z "$LATEST_RUN" ]; then
            # 如果没有找到offline-run目录，尝试查找普通run目录
            LATEST_RUN=$(find wandb -type d -name "run-*" | sort -r | head -n 1)
        fi

        if [ -n "$LATEST_RUN" ]; then
            echo "找到最新实验: $LATEST_RUN - $(date)" >> $SYNC_LOG
            
            # 检查是否已同步过
            if ! grep -q "^$LATEST_RUN$" "$SYNCED_RUNS_FILE"; then
                echo "尝试同步新实验 $LATEST_RUN - $(date)" >> $SYNC_LOG
                
                if $WANDB_CMD sync "$LATEST_RUN" >> $SYNC_LOG 2>&1; then
                    echo "$LATEST_RUN" >> "$SYNCED_RUNS_FILE"
                    echo "✓ 同步成功: $LATEST_RUN - $(date)" >> $SYNC_LOG
                    echo "最后同步时间: $(date), 状态: 成功" > monitor/last_sync_status.txt
                else
                    echo "✗ 同步失败: $LATEST_RUN - $(date)" >> $SYNC_LOG
                fi
            else
                # 已同步过，执行增量同步
                echo "尝试增量同步最新实验 $LATEST_RUN - $(date)" >> $SYNC_LOG
                if $WANDB_CMD sync --include-synced "$LATEST_RUN" >> $SYNC_LOG 2>&1; then
                    echo "✓ 增量同步成功: $LATEST_RUN - $(date)" >> $SYNC_LOG
                    echo "最后增量同步时间: $(date), 状态: 成功" > monitor/last_sync_status.txt
                else
                    echo "✗ 增量同步失败: $LATEST_RUN - $(date)" >> $SYNC_LOG
                fi
            fi
            
            # 显示最新实验信息
            echo "最新同步的实验: $LATEST_RUN" > monitor/latest_experiment.txt
            echo "同步时间: $(date)" >> monitor/latest_experiment.txt
        else
            echo "未找到任何实验目录 - $(date)" >> $SYNC_LOG
        fi
        
    else
        # 同步所有实验
        RUN_DIRS=($(find wandb -type d -name "*run-*"))
        for RUN_DIR in "${RUN_DIRS[@]}"; do
            # 首先检查是否在跳过列表中
            if grep -q "^$RUN_DIR$" "$SKIPPED_RUNS_FILE"; then
                echo "跳过已知问题运行: $RUN_DIR - $(date)" >> $SYNC_LOG
                continue
            fi

            # 检查是否为新目录
            if ! grep -q "^$RUN_DIR$" "$SYNCED_RUNS_FILE"; then
                echo "尝试同步新运行 $RUN_DIR - $(date)" >> $SYNC_LOG
                
                if $WANDB_CMD sync "$RUN_DIR" >> $SYNC_LOG 2>&1; then
                    echo "$RUN_DIR" >> "$SYNCED_RUNS_FILE"
                    echo "✓ 同步成功: $RUN_DIR - $(date)" >> $SYNC_LOG
                    echo "最后同步时间: $(date), 状态: 成功" > monitor/last_sync_status.txt
                fi
            else
                # 对已同步目录执行增量同步
                echo "尝试增量同步 $RUN_DIR - $(date)" >> $SYNC_LOG
                if $WANDB_CMD sync --include-synced "$RUN_DIR" >> $SYNC_LOG 2>&1; then
                    echo "✓ 增量同步成功: $RUN_DIR - $(date)" >> $SYNC_LOG
                    echo "最后增量同步时间: $(date), 状态: 成功" > monitor/last_sync_status.txt
                fi
            fi
        done
    fi
    
    # 3. 生成同步报告（可选）
    SYNCED_COUNT=$(wc -l < "$SYNCED_RUNS_FILE")
    echo "当前已同步的运行数: $SYNCED_COUNT" > monitor/sync_summary.txt
    echo "最后同步检查时间: $(date)" >> monitor/sync_summary.txt
    
    # 等待下一次同步
    echo "等待${SYNC_INTERVAL}秒进行下一次同步..." >> $SYNC_LOG
    sleep $SYNC_INTERVAL
done