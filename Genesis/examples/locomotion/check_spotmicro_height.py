"""
SpotMicroの初期関節角度で立たせたときの実際の高さを測定するスクリプト
train.pyと同じ設定を使用
物理シミュレーションを実行せずに、関節角度を設定した直後の高さを測定
"""
import genesis as gs
import torch
import numpy as np
import sys

# 初期化
gs.init()

# train.pyと同じデフォルト関節角度（get_spotmicro_cfgs()から）
default_joint_angles = {
    # 前左脚 (FL - Front Left)
    "motor_front_left_shoulder": 0.0,      # 肩関節（横方向の中立位置）
    "motor_front_left_leg": -0.8,          # 脚関節（約-46度）
    "foot_motor_front_left": 1.5,          # 足関節（約86度）
    # 前右脚 (FR - Front Right)
    "motor_front_right_shoulder": 0.0,     # 肩関節（横方向の中立位置）
    "motor_front_right_leg": -0.8,         # 脚関節（約-46度）
    "foot_motor_front_right": 1.5,         # 足関節（約86度）
    # 後左脚 (RL - Rear Left)
    "motor_rear_left_shoulder": 0.0,       # 肩関節（横方向の中立位置）
    "motor_rear_left_leg": -0.8,           # 脚関節（約-46度）
    "foot_motor_rear_left": 1.5,           # 足関節（約86度）
    # 後右脚 (RR - Rear Right)
    "motor_rear_right_shoulder": 0.0,      # 肩関節（横方向の中立位置）
    "motor_rear_right_leg": -0.8,          # 脚関節（約-46度）
    "foot_motor_rear_right": 1.5,          # 足関節（約86度）
}

# train.pyと同じジョイント名の順序（get_spotmicro_cfgs()から）
joint_names = [
    "motor_front_right_shoulder",
    "motor_front_right_leg",
    "foot_motor_front_right",
    "motor_front_left_shoulder",
    "motor_front_left_leg",
    "foot_motor_front_left",
    "motor_rear_right_shoulder",
    "motor_rear_right_leg",
    "foot_motor_rear_right",
    "motor_rear_left_shoulder",
    "motor_rear_left_leg",
    "foot_motor_rear_left",
]

# train.pyと同じ初期姿勢（get_spotmicro_cfgs()から）
base_init_pos = [0.0, 0.0, 0.30]  # train.pyと同じ初期高さ
base_init_quat = [0.0, 0.0, 0.0, 1.0]  # Z軸周りに180度回転（前後を入れ替える）

# train.pyと同じ目標高さ（get_spotmicro_cfgs()から）
base_height_target = 0.2325  # 目標ベース高さ（23.25cm）

# シーンを作成
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
    viewer_options=gs.options.ViewerOptions(
        max_FPS=60,
        camera_pos=(2.0, 0.0, 1.0),
        camera_lookat=(0.0, 0.0, 0.15),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=True,  # ビューアで確認できるようにする
)

# 地面を追加
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# ロボットを追加（train.pyと同じbase_init_posに配置）
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/workspace/Genesis/genesis/assets/urdf/spotmicro/urdf/spotmicro.urdf",
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

# 足のリンク名を取得（URDFから確認）
foot_link_names = [
    "front_left_foot_link",
    "front_right_foot_link",
    "rear_left_foot_link",
    "rear_right_foot_link",
]

# 足のリンクを取得
foot_links = [robot.get_link(name) for name in foot_link_names]

# 足の高さを測定
foot_heights = []
for foot_link in foot_links:
    foot_pos = foot_link.get_pos()
    foot_height = foot_pos[0, 2].item()
    foot_heights.append(foot_height)
min_foot_height = min(foot_heights)
max_foot_height = max(foot_heights)

print("\n" + "="*60)
print("測定結果（物理シミュレーション実行前）")
print("="*60)
print(f"初期関節角度設定時のベース高さ: {initial_height:.4f}m ({initial_height*100:.2f}cm)")
print(f"最低足高さ: {min_foot_height:.4f}m ({min_foot_height*100:.2f}cm)")
print(f"最高足高さ: {max_foot_height:.4f}m ({max_foot_height*100:.2f}cm)")
print(f"")
print(f"現在の設定 (train.pyから):")
print(f"  base_init_pos: {base_init_pos[2]:.3f}m ({base_init_pos[2]*100:.2f}cm)")
print(f"  base_height_target: {base_height_target:.3f}m ({base_height_target*100:.2f}cm)")
print(f"")
print(f"測定値と設定値の比較:")
print(f"  測定値（ベース高さ）: {initial_height*100:.2f}cm")
print(f"  base_init_pos: {base_init_pos[2]*100:.2f}cm (差: {abs(initial_height*100 - base_init_pos[2]*100):.2f}cm)")
print(f"  base_height_target: {base_height_target*100:.2f}cm (差: {abs(initial_height*100 - base_height_target*100):.2f}cm)")
print("="*60)
print("\nビューアで確認できます。Ctrl+Cで終了します。")
print("（物理シミュレーションは実行していません）")
sys.stdout.flush()

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
