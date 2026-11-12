#!/usr/bin/env bash
# Start the multi-seed DR data collection once the body-scale study pipeline has finished
# (both use the whole GPU; running them together would slow both down).
#   cd /workspace/vintix_go2 && setsid nohup bash dr_robots_multiseed_history/start_after_body_scale.sh \
#       > dr_robots_multiseed_history/logs/start_after.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."   # -> /workspace/vintix_go2
mkdir -p dr_robots_multiseed_history/logs
echo "[$(date '+%F %T')] waiting for robot_body_scale_zeroshot/pipeline.py to finish"
while pgrep -f "robot_body_scale_zeroshot/pipeline.py" > /dev/null; do sleep 300; done
if [ -f dr_robots_multiseed_history/STOP ]; then echo "STOP present, not starting"; exit 0; fi
echo "[$(date '+%F %T')] body-scale pipeline finished -> starting collect.py"
exec python -u dr_robots_multiseed_history/collect.py >> dr_robots_multiseed_history/logs/collect.log 2>&1
