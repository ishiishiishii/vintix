"""
UnitreeA1を目標高さ（base_height_target）で初期関節角度に設定し、
正面から見た画像を保存するスクリプト
"""
import genesis as gs
import torch
import numpy as np
import cv2
import os
import sys

# 初期化
gs.init()

# train.pyと同じデフォルト関節角度（get_unitreea1_cfgs()から）
default_joint_angles = {
    # 前脚: やや低めの姿勢で安定性を確保
    "FR_hip_joint": 0.0,      # 中間位置（リミット: ±0.80）
    "FR_thigh_joint": 0.8,    # 約46度前向き（リミット: -1.05 to 4.19）
    "FR_calf_joint": -1.5,    # 約-86度（リミット: -2.70 to -0.92）
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    # 後脚: 前脚よりやや高めで推進力を確保
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 1.0,    # 約57度（リミット: -1.05 to 4.19）
    "RR_calf_joint": -1.5,    # 約-86度
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
}

# train.pyと同じジョイント名の順序（get_unitreea1_cfgs()から）
joint_names = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# train.pyと同じ目標高さ（get_unitreea1_cfgs()から）
base_height_target = 0.27  # 目標高さ0.27m（27cm）
base_init_quat = [1.0, 0.0, 0.0, 0.0]  # train.pyと同じ

# 保存先ディレクトリ
save_dir = "/workspace/Genesis/genesis/imgs"
os.makedirs(save_dir, exist_ok=True)

# シーンを作成（ビューアなし、ヘッドレスモード）
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
    vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=False,  # ビューアなしで実行
)

# 地面を追加
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# ロボットを目標高さで配置
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/workspace/Genesis/genesis/assets/urdf/unitree_a1/urdf/a1.urdf",
        pos=[0.0, 0.0, base_height_target],  # 目標高さで配置
        quat=base_init_quat,
    ),
)

# カメラを追加（斜め右上から見た画角、ロボットを拡大）
# 斜め右上から見る: x=前方、y=右側、z=上方
camera_x = 1.8  # 前方距離（メートル）- 短くして拡大
camera_y = 1.2  # 右側距離（メートル）- 斜めから見る
camera_z = 0.5  # 上方距離（メートル）- 上から見下ろす

cam = scene.add_camera(
    res=(1280, 960),  # 高解像度
    pos=(camera_x, camera_y, camera_z),  # 斜め右上から
    lookat=(0.0, 0.0, base_height_target),  # ロボットの中心を見る
    fov=40,  # 視野角
    GUI=False,  # GUIなし
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

# 関節角度を設定
robot.set_dofs_position(
    position=default_dof_pos.unsqueeze(0),
    dofs_idx_local=motors_dof_idx,
    zero_velocity=True,
    envs_idx=[0],
)

# 実際のベース高さを測定
base_pos = robot.get_pos()
actual_base_height = base_pos[0, 2].item()

print(f"目標高さ: {base_height_target*100:.2f}cm")
print(f"実際のベース高さ: {actual_base_height*100:.2f}cm")
print(f"差: {abs(actual_base_height - base_height_target)*100:.2f}cm")

# シーンを更新（カメラが正しく設定されるように）
scene._visualizer.update(force=True, auto=True)

# 画像をレンダリング
print("\n画像をレンダリング中...")
rgb, depth, segmentation, normal = cam.render(rgb=True)

# RGB画像を保存用に変換
if rgb is not None:
    # numpy配列に変換（torch tensorの場合はcpu().numpy()で変換）
    if hasattr(rgb, 'cpu'):
        rgb_array = rgb.cpu().numpy()
    elif hasattr(rgb, 'numpy'):
        rgb_array = rgb.numpy()
    else:
        rgb_array = np.array(rgb)
    
    # 多次元配列の場合、最初の環境の画像を取得
    if len(rgb_array.shape) == 4:  # (n_envs, height, width, channels)
        rgb_array = rgb_array[0]
    elif len(rgb_array.shape) == 3:  # (height, width, channels)
        rgb_array = rgb_array
    
    # 値の範囲を確認して正規化
    if rgb_array.max() <= 1.0:
        rgb_array = (rgb_array * 255).astype(np.uint8)
    else:
        rgb_array = rgb_array.astype(np.uint8)
    
    # RGBからBGRに変換（OpenCVはBGR形式を使用）
    # GenesisのカメラはRGB形式で返すので、BGRに変換
    if rgb_array.shape[2] == 3:  # RGB画像の場合
        rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    else:
        rgb_bgr = rgb_array
    
    # 画像を保存
    save_path = os.path.join(save_dir, "unitreea1_target_height_diagonal.png")
    success = cv2.imwrite(save_path, rgb_bgr)
    if success:
        print(f"画像を保存しました: {save_path}")
        print(f"画像サイズ: {rgb_bgr.shape}")
    else:
        print(f"エラー: 画像の保存に失敗しました: {save_path}")
else:
    print("エラー: 画像のレンダリングに失敗しました")

print("\n完了しました。")
