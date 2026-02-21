"""
Minicheetahの初期関節角度で立たせたときの実際の高さを測定するスクリプト
train.pyと同じ設定を使用
"""
import genesis as gs
import torch
import numpy as np

# 初期化
gs.init()

# train.pyと同じデフォルト関節角度
default_joint_angles = {
    "torso_to_abduct_fl_j": 0.0,
    "torso_to_abduct_fr_j": 0.0,
    "torso_to_abduct_hr_j": 0.0,
    "torso_to_abduct_hl_j": 0.0,
    "abduct_fl_to_thigh_fl_j": -0.8,
    "abduct_fr_to_thigh_fr_j": -0.8,
    "abduct_hr_to_thigh_hr_j": -0.8,
    "abduct_hl_to_thigh_hl_j": -0.8,
    "thigh_fl_to_knee_fl_j": 1.5,
    "thigh_fr_to_knee_fr_j": 1.5,
    "thigh_hr_to_knee_hr_j": 1.5,
    "thigh_hl_to_knee_hl_j": 1.5,
}

# train.pyと同じジョイント名の順序
joint_names = [
    "torso_to_abduct_fr_j",
    "abduct_fr_to_thigh_fr_j",
    "thigh_fr_to_knee_fr_j",
    "torso_to_abduct_fl_j",
    "abduct_fl_to_thigh_fl_j",
    "thigh_fl_to_knee_fl_j",
    "torso_to_abduct_hr_j",
    "abduct_hr_to_thigh_hr_j",
    "thigh_hr_to_knee_hr_j",
    "torso_to_abduct_hl_j",
    "abduct_hl_to_thigh_hl_j",
    "thigh_hl_to_knee_hl_j",
]

# train.pyと同じ初期姿勢
base_init_pos = [0.0, 0.0, 0.30]  # train.pyと同じ
base_init_quat = [1.0, 0.0, 0.0, 0.0]  # train.pyと同じ

# シーンを作成
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
    viewer_options=gs.options.ViewerOptions(
        max_FPS=60,
        camera_pos=(3.0, 0.0, 2.0),
        camera_lookat=(0.0, 0.0, 0.3),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=True,
)

# 地面を追加
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# ロボットを追加（train.pyと同じbase_init_posに配置）
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/workspace/Genesis/genesis/assets/urdf/mini_cheetah/urdf/mini_cheetah.urdf",
        pos=base_init_pos,  # train.pyと同じ初期位置
        quat=base_init_quat,  # train.pyと同じ初期向き
    ),
)

# シーンをビルド
scene.build(n_envs=1)

# ジョイントインデックスを取得
motors_dof_idx = [robot.get_joint(name).dof_start for name in joint_names]

# train.pyと同じデフォルト関節角度を設定
default_dof_pos = torch.tensor(
    [default_joint_angles[name] for name in joint_names],
    device=gs.device,
    dtype=gs.tc_float,
)

# 関節角度を設定（train.pyと同じ）
robot.set_dofs_position(
    position=default_dof_pos.unsqueeze(0),
    dofs_idx_local=motors_dof_idx,
    zero_velocity=True,
    envs_idx=[0],
)

# 物理シミュレーションを実行せずに、関節角度を設定した直後のベース高さを測定
base_pos = robot.get_pos()
initial_height = base_pos[0, 2].item()

# 数値を確実に出力するために、sys.stdout.flush()を使用
import sys

print("\n" + "="*60)
print("測定結果（物理シミュレーション実行前）")
print("="*60)
print(f"初期関節角度設定時のベース高さ: {initial_height:.4f}m ({initial_height*100:.2f}cm)")
print(f"")
print(f"現在の設定 (train.pyから):")
print(f"  base_init_pos: {base_init_pos[2]:.2f}m ({base_init_pos[2]*100:.2f}cm)")
print(f"  base_height_target: 0.22m (22cm)")
print(f"")
print(f"測定値と設定値の比較:")
print(f"  測定値: {initial_height*100:.2f}cm")
print(f"  base_init_pos: {base_init_pos[2]*100:.2f}cm (差: {abs(initial_height*100 - base_init_pos[2]*100):.2f}cm)")
print(f"  base_height_target: 22.00cm (差: {abs(initial_height*100 - 22.0):.2f}cm)")
print("="*60)
print("\nビューアで確認できます。Ctrl+Cで終了します。")
print("（物理シミュレーションは実行していません）")
sys.stdout.flush()  # 出力を確実にフラッシュ

# ビューアを表示し続ける（物理シミュレーションは実行しない）
try:
    import time
    # ビューアを一度更新
    scene._visualizer.update(force=True, auto=True)
    # ビューアを表示し続ける（物理シミュレーションなし）
    while True:
        time.sleep(0.1)  # ビューアを維持
except KeyboardInterrupt:
    print("\n終了します。")

