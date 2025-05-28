#!/bin/bash

# 配置参数
SYNC_INTERVAL=30
SYNC_LOG="multimonitor/auto_sync.log"
SYNCED_RUNS_FILE="multimonitor/synced_runs.txt"
SKIPPED_RUNS_FILE="multimonitor/skipped_runs.txt"
REMOTE_PATH="/home/zengh/2025/Grokking/grokkingRe"

# 动态获取运行中作业的节点信息
get_active_nodes() {
    # 使用j命令获取当前用户的作业信息
    j 2>/dev/null | grep "zengh" | grep " R " | awk '{
        # 提取节点信息（最后一列，如 node28/0*24）
        split($NF, parts, "/")
        if (parts[1] != "--") {
            print parts[1] ".cluster"
        }
    }' | sort -u > /tmp/active_nodes.txt
    
    if [ -s /tmp/active_nodes.txt ]; then
        ACTIVE_NODES=($(cat /tmp/active_nodes.txt))
        echo "动态检测到活跃节点: ${ACTIVE_NODES[@]}" >> $SYNC_LOG
    else
        ACTIVE_NODES=()
        echo "未检测到运行中的作业节点" >> $SYNC_LOG
    fi
}

# 检查特定节点是否有训练任务
check_node_training_task() {
    local node=$1
    local node_short=$(echo $node | cut -d'.' -f1)  # 去掉.cluster后缀
    
    # 检查SSH连接
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes $node "echo 'test'" >/dev/null 2>&1; then
        echo "⚠ 无法连接到节点 $node" >> $SYNC_LOG
        return 1
    fi
    
    # 检查是否有need_sync.flag文件
    if ssh $node "[ -f $REMOTE_PATH/need_sync.flag ]"; then
        echo "✓ 节点 $node 有训练任务运行" >> $SYNC_LOG
        return 0
    else
        echo "- 节点 $node 无训练任务" >> $SYNC_LOG
        return 1
    fi
}

SYNC_MODE="all"
if [ "$1" = "--latest" ]; then
    SYNC_MODE="latest"
    echo "仅同步最新实验模式已启用" >> $SYNC_LOG
fi

# 创建必要的目录
mkdir -p multimonitor
touch "$SYNCED_RUNS_FILE"
touch "$SKIPPED_RUNS_FILE"

# 记录脚本启动
echo "启动动态多节点自动同步 - $(date)" > $SYNC_LOG

# 激活Python环境
echo "激活Python环境..." >> $SYNC_LOG
source ~/.bashrc
conda activate base
echo "Python环境激活状态: $?" >> $SYNC_LOG

# 检查wandb
if python -c "import wandb" 2>/dev/null; then
    echo "wandb Python包可用" >> $SYNC_LOG
    WANDB_CMD="python -m wandb"
else
    echo "错误: wandb Python包不可用" >> $SYNC_LOG
    exit 1
fi
echo "wandb命令设置为: $WANDB_CMD" >> $SYNC_LOG

# 初始化状态跟踪
declare -A NODE_WAS_RUNNING

# 主循环
while true; do
    echo "=== 开始新一轮动态节点检查 - $(date) ===" >> $SYNC_LOG
    
    # 1. 动态获取活跃节点
    get_active_nodes
    
    if [ ${#ACTIVE_NODES[@]} -eq 0 ]; then
        echo "当前无运行中的作业，等待新作业..." >> $SYNC_LOG
        sleep $SYNC_INTERVAL
        continue
    fi
    
    ANY_TASK_RUNNING=false
    
    # 2. 检查每个活跃节点
    for node in "${ACTIVE_NODES[@]}"; do
        echo "检查节点 $node - $(date)" >> $SYNC_LOG
        
        if check_node_training_task "$node"; then
            ANY_TASK_RUNNING=true
            NODE_WAS_RUNNING[$node]=true
            
            echo "从节点 $node 同步数据..." >> $SYNC_LOG
            
            # # 创建节点特定的wandb目录
            # mkdir -p "wandb_${node}"
            
            # # 从计算节点同步wandb目录
            # rsync -az --exclude="*.tmp" $node:$REMOTE_PATH/wandb/ wandb_${node}/ >> $SYNC_LOG 2>&1
            
            # # 合并到主wandb目录
            # rsync -az wandb_${node}/ wandb/ >> $SYNC_LOG 2>&1
            
            # 直接同步到主wandb目录，不创建节点特定目录
            rsync -az --exclude="*.tmp" $node:$REMOTE_PATH/wandb/ wandb/ >> $SYNC_LOG 2>&1

            # 同步训练日志
            rsync -az $node:$REMOTE_PATH/training.log multimonitor/training_${node}.log >> $SYNC_LOG 2>&1
            rsync -az $node:$REMOTE_PATH/debug.log multimonitor/debug_${node}.log >> $SYNC_LOG 2>&1
            
            echo "✓ 节点 $node 数据同步完成" >> $SYNC_LOG
            
        else
            # 检查节点是否从运行状态变为停止状态
            if [ "${NODE_WAS_RUNNING[$node]}" = true ]; then
                echo "节点 $node 任务已停止，执行最后一次同步..." >> $SYNC_LOG
                
                # 最后一次同步
                # rsync -az --exclude="*.tmp" $node:$REMOTE_PATH/wandb/ wandb_${node}/ >> $SYNC_LOG 2>&1
                # rsync -az wandb_${node}/ wandb/ >> $SYNC_LOG 2>&1
                # 最后一次同步
                rsync -az --exclude="*.tmp" $node:$REMOTE_PATH/wandb/ wandb/ >> $SYNC_LOG 2>&1
                rsync -az $node:$REMOTE_PATH/training.log multimonitor/training_${node}_final.log >> $SYNC_LOG 2>&1
                
                # 标记该节点不再运行
                NODE_WAS_RUNNING[$node]=false
                echo "✓ 节点 $node 最终同步完成" >> $SYNC_LOG
            fi
        fi
    done
    
    # 3. 如果有任务运行，向wandb同步数据
    if [ "$ANY_TASK_RUNNING" = true ]; then
        echo "开始向wandb同步数据 - $(date)" >> $SYNC_LOG

        # 预处理已删除的运行ID（保持原有逻辑）
        echo "检查历史日志中的已删除运行..." >> $SYNC_LOG
        grep -E "previously created and deleted" "$SYNC_LOG" | grep -E "run [a-z0-9]+" | \
            awk '{print $5}' | sort -u > /tmp/deleted_run_ids.txt

        if [ -s /tmp/deleted_run_ids.txt ]; then
            echo "从历史日志中发现已删除的运行ID:" >> $SYNC_LOG
            cat /tmp/deleted_run_ids.txt >> $SYNC_LOG
            
            while read -r run_id; do
                id_only=$(echo "$run_id" | grep -oE "[a-z0-9]+")
                matching_dir=$(find wandb -type d -name "*$id_only*" | head -n 1)
                if [ -n "$matching_dir" ] && ! grep -q "^$matching_dir$" "$SKIPPED_RUNS_FILE"; then
                    echo "$matching_dir" >> "$SKIPPED_RUNS_FILE"
                    echo "添加到跳过列表: $matching_dir (ID: $id_only)" >> $SYNC_LOG
                fi
            done < /tmp/deleted_run_ids.txt
        fi

        # 同步逻辑（根据模式）
        if [ "$SYNC_MODE" = "latest" ]; then
            # 最新实验同步逻辑（保持原有）
            LATEST_RUN=$(find wandb -type d -name "offline-run-*" | sort -r | head -n 1)
            if [ -z "$LATEST_RUN" ]; then
                LATEST_RUN=$(find wandb -type d -name "run-*" | sort -r | head -n 1)
            fi

            if [ -n "$LATEST_RUN" ]; then
                echo "找到最新实验: $LATEST_RUN - $(date)" >> $SYNC_LOG
                
                if ! grep -q "^$LATEST_RUN$" "$SYNCED_RUNS_FILE"; then
                    echo "尝试同步新实验 $LATEST_RUN - $(date)" >> $SYNC_LOG
                    if $WANDB_CMD sync "$LATEST_RUN" >> $SYNC_LOG 2>&1; then
                        echo "$LATEST_RUN" >> "$SYNCED_RUNS_FILE"
                        echo "✓ 同步成功: $LATEST_RUN - $(date)" >> $SYNC_LOG
                    fi
                else
                    echo "尝试增量同步最新实验 $LATEST_RUN - $(date)" >> $SYNC_LOG
                    if $WANDB_CMD sync --include-synced "$LATEST_RUN" >> $SYNC_LOG 2>&1; then
                        echo "✓ 增量同步成功: $LATEST_RUN - $(date)" >> $SYNC_LOG
                    fi
                fi
            fi
        else
            # 同步所有实验（保持原有逻辑）
            RUN_DIRS=($(find wandb -type d -name "*run-*"))
            for RUN_DIR in "${RUN_DIRS[@]}"; do
                if grep -q "^$RUN_DIR$" "$SKIPPED_RUNS_FILE"; then
                    echo "跳过已知问题运行: $RUN_DIR - $(date)" >> $SYNC_LOG
                    continue
                fi

                if ! grep -q "^$RUN_DIR$" "$SYNCED_RUNS_FILE"; then
                    echo "尝试同步新运行 $RUN_DIR - $(date)" >> $SYNC_LOG
                    if $WANDB_CMD sync "$RUN_DIR" >> $SYNC_LOG 2>&1; then
                        echo "$RUN_DIR" >> "$SYNCED_RUNS_FILE"
                        echo "✓ 同步成功: $RUN_DIR - $(date)" >> $SYNC_LOG
                    fi
                else
                    echo "尝试增量同步 $RUN_DIR - $(date)" >> $SYNC_LOG
                    if $WANDB_CMD sync --include-synced "$RUN_DIR" >> $SYNC_LOG 2>&1; then
                        echo "✓ 增量同步成功: $RUN_DIR - $(date)" >> $SYNC_LOG
                    fi
                fi
            done
        fi
    fi
    
    # 4. 生成多节点同步报告
    echo "活跃节点数: ${#ACTIVE_NODES[@]}" > multimonitor/sync_summary.txt
    echo "活跃节点: ${ACTIVE_NODES[@]}" >> multimonitor/sync_summary.txt
    SYNCED_COUNT=$(wc -l < "$SYNCED_RUNS_FILE")
    echo "已同步运行数: $SYNCED_COUNT" >> multimonitor/sync_summary.txt
    echo "最后检查时间: $(date)" >> multimonitor/sync_summary.txt
    
    echo "等待${SYNC_INTERVAL}秒进行下一次检查..." >> $SYNC_LOG
    sleep $SYNC_INTERVAL
done