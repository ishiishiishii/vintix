"""
Laikagoの回転の仕組みを調べるスクリプト
ターミナルから数字を入力して回転を切り替え可能
"""
import genesis as gs
import torch
import numpy as np
import sys
import threading
import time

# 初期化
gs.init(backend=gs.cuda)

# デフォルト関節角度
default_joint_angles = {
    "FR_hip_motor_2_chassis_joint": 0.0,
    "FR_upper_leg_2_hip_motor_joint": 0.0,
    "FR_lower_leg_2_upper_leg_joint": -0.75,
    "FL_hip_motor_2_chassis_joint": 0.0,
    "FL_upper_leg_2_hip_motor_joint": 0.0,
    "FL_lower_leg_2_upper_leg_joint": -0.75,
    "RR_hip_motor_2_chassis_joint": 0.0,
    "RR_upper_leg_2_hip_motor_joint": 0.0,
    "RR_lower_leg_2_upper_leg_joint": -0.75,
    "RL_hip_motor_2_chassis_joint": 0.0,
    "RL_upper_leg_2_hip_motor_joint": 0.0,
    "RL_lower_leg_2_upper_leg_joint": -0.75
}

joint_names = [
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint"
]

base_init_pos = [0.0, 0.0, 0.50]

# 複数の回転を試す（数字で選択可能）
test_rotations = {
    "1": {"name": "回転なし", "quat": [1.0, 0.0, 0.0, 0.0]},
    "2": {"name": "Y軸90度", "quat": [0.7071068, 0.0, 0.7071068, 0.0]},
    "3": {"name": "Y軸90度→X軸90度（固定座標系）", "quat": [0.5, 0.5, 0.5, 0.5]},
    "4": {"name": "Y軸90度→X軸90度（符号反転）", "quat": [-0.5, -0.5, -0.5, -0.5]},
    "5": {"name": "Z軸90度", "quat": [0.7071068, 0.0, 0.0, 0.7071068]},
    "6": {"name": "X軸90度", "quat": [0.7071068, 0.7071068, 0.0, 0.0]},
}

# デフォルトは回転なし
current_rotation_key = "1"
base_init_quat = test_rotations[current_rotation_key]["quat"]

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
    show_viewer=True,
)

# 地面を追加
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# ロボットを追加
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/workspace/Genesis/genesis/assets/urdf/laikago/urdf/laikago.urdf",
        pos=base_init_pos,
        quat=base_init_quat,
    ),
)

# シーンをビルド
scene.build(n_envs=1)

# ジョイントインデックスを取得
motors_dof_idx = [robot.get_joint(name).dof_start for name in joint_names]

# デフォルト関節角度を設定
default_dof_pos = torch.tensor(
    [default_joint_angles[name] for name in joint_names],
    device=gs.device,
    dtype=gs.tc_float,
)

robot.set_dofs_position(
    position=default_dof_pos.unsqueeze(0),
    dofs_idx_local=motors_dof_idx,
    zero_velocity=True,
    envs_idx=[0],
)

# ベース高さを測定
base_pos = robot.get_pos()
initial_height = base_pos[0, 2].item()

print(f"\n{'='*60}")
print(f"Laikago回転テストツール")
print(f"{'='*60}")
print(f"ベース高さ: {initial_height:.4f}m ({initial_height*100:.2f}cm)")
print(f"\n現在の回転: {test_rotations[current_rotation_key]['name']}")
print(f"クォータニオン [w, x, y, z]: {base_init_quat}")
print(f"\n使用可能な回転（数字を入力して切り替え）:")
for key, value in sorted(test_rotations.items()):
    marker = " <-- 現在" if key == current_rotation_key else ""
    print(f"  {key}: {value['name']}{marker}")
print(f"\n操作方法:")
print(f"  - 数字(1-6)を入力して回転を切り替え")
print(f"  - 'q'または'quit'で終了")
print(f"  - Ctrl+Cでも終了可能")
print(f"{'='*60}\n")
sys.stdout.flush()

# 回転変更用のロック
rotation_lock = threading.Lock()
current_quat = torch.tensor(base_init_quat, device=gs.device, dtype=gs.tc_float)

def apply_rotation(rotation_key):
    """指定された回転を適用"""
    global current_quat, current_rotation_key
    if rotation_key in test_rotations:
        with rotation_lock:
            current_rotation_key = rotation_key
            rotation_info = test_rotations[rotation_key]
            current_quat = torch.tensor(rotation_info["quat"], device=gs.device, dtype=gs.tc_float)
            robot.set_quat(current_quat.unsqueeze(0), envs_idx=[0], zero_velocity=True)
            print(f"\n回転を変更しました: {rotation_info['name']}")
            print(f"クォータニオン [w, x, y, z]: {rotation_info['quat']}")
            sys.stdout.flush()
            return True
    return False

def input_thread():
    """標準入力を受け取るスレッド"""
    while True:
        try:
            user_input = input().strip().lower()
            if user_input in ['q', 'quit', 'exit']:
                print("\n終了します...")
                sys.exit(0)
            elif user_input in test_rotations:
                apply_rotation(user_input)
            else:
                print(f"無効な入力: '{user_input}'. 1-6の数字、または'q'を入力してください。")
                sys.stdout.flush()
        except EOFError:
            break
        except Exception as e:
            print(f"入力エラー: {e}")
            sys.stdout.flush()

# 入力スレッドを開始
input_thread_obj = threading.Thread(target=input_thread, daemon=True)
input_thread_obj.start()

# ビューアを表示し続ける
try:
    scene._visualizer.update(force=True, auto=True)
    while True:
        scene._visualizer.update(force=True, auto=True)
        time.sleep(0.05)  # ビューアの更新頻度
except KeyboardInterrupt:
    print("\n終了します。")

