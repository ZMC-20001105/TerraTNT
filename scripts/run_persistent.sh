#!/bin/bash
# 持久化运行脚本 - 使用nohup保证断开SSH后进程继续运行
# 用法: ./run_persistent.sh <command> <log_name>
# 示例: ./run_persistent.sh "python train.py" training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

if [ -z "$1" ]; then
    echo "用法: $0 <command> [log_name]"
    echo "示例: $0 'python scripts/train_all_baselines.py' training"
    exit 1
fi

COMMAND="$1"
LOG_NAME="${2:-task}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${LOG_NAME}_${TIMESTAMP}.log"

echo "=========================================="
echo "启动持久化任务"
echo "命令: $COMMAND"
echo "日志: $LOG_FILE"
echo "=========================================="

cd "$PROJECT_DIR"
nohup bash -c "$COMMAND" > "$LOG_FILE" 2>&1 &
PID=$!
disown $PID

echo "进程已启动, PID: $PID"
echo "查看日志: tail -f $LOG_FILE"
echo "查看进程: ps aux | grep $PID"
