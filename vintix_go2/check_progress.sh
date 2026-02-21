#!/bin/bash
echo '=== AD データ収集の進捗 ==='
tail -3 ad_collect_final.log | grep 'Collecting AD data'
echo ''
echo '=== 最近のエピソード統計 ==='
tail -30 ad_collect_final.log | grep 'Episode' | tail -5
echo ''
echo '=== ファイルサイズ ==='
ls -lh data/go2_trajectories/go2_ad_final/ 2>/dev/null || echo 'まだ作成中...'
