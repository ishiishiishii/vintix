"""
Laikagoの初期関節角度で立たせたときの実際の高さを測定するスクリプト
train.pyと同じ設定を使用
"""
import genesis as gs
import torch
import numpy as np
import sys

# 初期化
gs.init()

# train.pyと同じデフォルト関節角度（get_laikago_cfgs()から）
default_joint_angles = {
    # Front Right leg (FR) - 1st
    "FR_hip_motor_2_chassis_joint": 0.0,      # Chassis: 0
    "FR_upper_leg_2_hip_motor_joint": 0.0,    # Upperleg: 0
    "FR_lower_leg_2_upper_leg_joint": -0.75,  # Lowerleg: -0.75
    # Front Left leg (FL) - 2nd
    "FL_hip_motor_2_chassis_joint": 0.0,      # Chassis: 0
    "FL_upper_leg_2_hip_motor_joint": 0.0,   # Upperleg: 0
    "FL_lower_leg_2_upper_leg_joint": -0.75, # Lowerleg: -0.75
    # Rear Right leg (RR) - 3rd
    "RR_hip_motor_2_chassis_joint": 0.0,      # Chassis: 0
    "RR_upper_leg_2_hip_motor_joint": 0.0,   # Upperleg: 0
    "RR_lower_leg_2_upper_leg_joint": -0.75, # Lowerleg: -0.75
    # Rear Left leg (RL) - 4th
    "RL_hip_motor_2_chassis_joint": 0.0,      # Chassis: 0
    "RL_upper_leg_2_hip_motor_joint": 0.0,   # Upperleg: 0
    "RL_lower_leg_2_upper_leg_joint": -0.75  # Lowerleg: -0.75
}

# train.pyと同じジョイント名の順序（get_laikago_cfgs()から）
joint_names = [
    # Front Right leg (FR) - 1st
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint", 
    "FR_lower_leg_2_upper_leg_joint",
    # Front Left leg (FL) - 2nd
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint", 
    # Rear Right leg (RR) - 3rd
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    # Rear Left leg (RL) - 4th
    "RL_hip_motor_2_chassis_joint", 
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint"
]

# train.pyと同じ初期姿勢（get_laikago_cfgs()から）
# 固定座標系での回転: Y軸90度→X軸90度（R_x @ R_y の順序）
# transform_quat_by_quat(v, u)は R_u @ R_v を計算するため、固定座標系での回転
base_init_pos = [0.0, 0.0, 0.50]  # Laikagoのスタンディング高さ（50cm）
base_init_quat = [0.5, 0.5, 0.5, 0.5]  # Y軸90度→X軸90度（固定座標系、正面=X軸、足=-Z軸）

# train.pyと同じ目標高さ（get_laikago_cfgs()から）
base_height_target = 0.50  # 目標高さ0.50m（50cm）

# シーンを作成
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
    viewer_options=gs.options.ViewerOptions(
        max_FPS=60,
        camera_pos=(3.0, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=True,  # ビューアありで実行
)

# 地面を追加
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# ロボットを追加（train.pyと同じbase_init_posに配置）
# 注意: gs.morphs.URDFのquatパラメータはURDFの元の回転と合成されるため、
# 絶対的な回転を設定するには、最初は回転なしで追加し、後でset_quat()を使用
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/workspace/Genesis/genesis/assets/urdf/laikago/urdf/laikago.urdf",
        pos=base_init_pos,  # train.pyと同じ初期位置
        quat=[1.0, 0.0, 0.0, 0.0],  # 回転なしで追加（後でset_quat()で設定）
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

# train.pyと同じ回転を設定（set_quat()は絶対的な回転を設定）
# check_laikago_rotation.pyと同じ方法で、絶対的な回転を設定
robot.set_quat(
    torch.tensor(base_init_quat, device=gs.device, dtype=gs.tc_float).unsqueeze(0),
    envs_idx=[0],
    zero_velocity=True,
)

# 物理シミュレーションを実行せずに、関節角度を設定した直後のベース高さを測定
base_pos = robot.get_pos()
initial_height = base_pos[0, 2].item()

# 数値を確実に出力するために、sys.stdout.flush()を使用
print("\n" + "="*60)
print("測定結果（物理シミュレーション実行前）")
print("="*60)
print(f"初期関節角度設定時のベース高さ: {initial_height:.4f}m ({initial_height*100:.2f}cm)")
print(f"")
print(f"現在の設定 (train.pyから):")
print(f"  base_init_pos: {base_init_pos[2]:.2f}m ({base_init_pos[2]*100:.2f}cm)")
print(f"  base_height_target: {base_height_target:.2f}m ({base_height_target*100:.2f}cm)")
print(f"  base_init_quat: {base_init_quat} (Z軸周りに-90度回転)")
print(f"")
print(f"測定値と設定値の比較:")
print(f"  測定値: {initial_height*100:.2f}cm")
print(f"  base_init_pos: {base_init_pos[2]*100:.2f}cm (差: {abs(initial_height*100 - base_init_pos[2]*100):.2f}cm)")
print(f"  base_height_target: {base_height_target*100:.2f}cm (差: {abs(initial_height*100 - base_height_target*100):.2f}cm)")
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

