#!/bin/bash
# 带时间戳运行 compute_similarity_whole.py（不改动原脚本）
# 每行日志加 [HH:MM:SS] 前缀，实时打印并写入 run_all_*.log
# 用法: bash run_with_timestamps.sh [--dry-run] [--models m1,m2] [--cases 1,2]

cd "$(dirname "$0")"
LOG="run_all_$(date +%Y%m%d_%H%M%S).log"
echo "日志文件: $LOG"
python -u compute_similarity_whole.py "$@" 2>&1 | while IFS= read -r line; do
    ts=$(date +%H:%M:%S)
    printf '[%s] %s\n' "$ts" "$line"
    printf '[%s] %s\n' "$ts" "$line" >> "$LOG"
done
rc=${PIPESTATUS[0]}
ts=$(date +%H:%M:%S)
printf '[%s] 脚本退出码: %d\n' "$ts" "$rc" | tee -a "$LOG"
exit $rc
