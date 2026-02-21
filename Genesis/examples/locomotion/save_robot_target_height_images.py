"""
複数のロボット（UnitreeA1, Minicheetah, Go1, Go2）を目標高さ（base_height_target）で
初期関節角度に設定し、斜め右上から見た画像を保存するスクリプト
"""
import genesis as gs
import torch
import numpy as np
import cv2
import os
import sys
from PIL import Image

# 初期化
gs.init()

# ロボット設定の定義
ROBOT_CONFIGS = {
    "unitreea1": {
        "urdf_path": "/workspace/Genesis/genesis/assets/urdf/unitree_a1/urdf/a1.urdf",
        "base_height_target": 0.27,  # 27cm
        "default_joint_angles": {
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
    },
    "minicheetah": {
        "urdf_path": "/workspace/Genesis/genesis/assets/urdf/mini_cheetah/urdf/mini_cheetah.urdf",
        "base_height_target": 0.27,  # 27cm
        "default_joint_angles": {
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
        },
        "joint_names": [
            "torso_to_abduct_fr_j", "abduct_fr_to_thigh_fr_j", "thigh_fr_to_knee_fr_j",
            "torso_to_abduct_fl_j", "abduct_fl_to_thigh_fl_j", "thigh_fl_to_knee_fl_j",
            "torso_to_abduct_hr_j", "abduct_hr_to_thigh_hr_j", "thigh_hr_to_knee_hr_j",
            "torso_to_abduct_hl_j", "abduct_hl_to_thigh_hl_j", "thigh_hl_to_knee_hl_j",
        ],
    },
    "go1": {
        "urdf_path": "/workspace/Genesis/genesis/assets/urdf/go1/urdf/go1.urdf",
        "base_height_target": 0.30,  # 30cm
        "default_joint_angles": {
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.6,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.6,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.6,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.6,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
    },
    "go2": {
        "urdf_path": "/workspace/Genesis/genesis/assets/urdf/go2/urdf/go2.urdf",
        "base_height_target": 0.3,  # 30cm
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
    },
}

# 保存先ディレクトリ（上の上の階層のGenesis/imgs）
save_dir = "/workspace/Genesis/imgs"
os.makedirs(save_dir, exist_ok=True)

base_init_quat = [1.0, 0.0, 0.0, 0.0]

def save_robot_image(robot_name, config):
    """指定されたロボットの画像を保存する"""
    print(f"\n{'='*60}")
    print(f"処理中: {robot_name}")
    print(f"{'='*60}")
    
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
    base_height_target = config["base_height_target"]
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=config["urdf_path"],
            pos=[0.0, 0.0, base_height_target],  # 目標高さで配置
            quat=base_init_quat,
        ),
    )
    
    # カメラを追加（斜め右上から見た画角、ロボットを拡大）
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
    motors_dof_idx = [robot.get_joint(name).dof_start for name in config["joint_names"]]
    
    # デフォルト関節角度を設定
    default_dof_pos = torch.tensor(
        [config["default_joint_angles"][name] for name in config["joint_names"]],
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
    print("画像をレンダリング中...")
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
        if rgb_array.shape[2] == 3:  # RGB画像の場合
            rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        else:
            rgb_bgr = rgb_array
        
        # 画像をPNG形式で保存
        save_path_png = os.path.join(save_dir, f"{robot_name}_target_height_diagonal.png")
        success = cv2.imwrite(save_path_png, rgb_bgr)
        if success:
            print(f"PNG画像を保存しました: {save_path_png}")
            print(f"画像サイズ: {rgb_bgr.shape}")
            
            # PDF形式でも保存
            try:
                # RGB形式に戻す（PILはRGB形式を使用）
                rgb_for_pil = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_for_pil)
                save_path_pdf = os.path.join(save_dir, f"{robot_name}_target_height_diagonal.pdf")
                pil_image.save(save_path_pdf, "PDF", resolution=100.0, quality=95)
                print(f"PDF画像を保存しました: {save_path_pdf}")
            except Exception as e:
                print(f"警告: PDF保存に失敗しました: {e}")
        else:
            print(f"エラー: 画像の保存に失敗しました: {save_path_png}")
    else:
        print("エラー: 画像のレンダリングに失敗しました")
    
    # シーンをクリーンアップ
    del scene
    del cam
    del robot

# すべてのロボットの画像を保存
print("ロボット画像の保存を開始します...")
for robot_name, config in ROBOT_CONFIGS.items():
    try:
        save_robot_image(robot_name, config)
    except Exception as e:
        print(f"エラー: {robot_name}の画像保存に失敗しました: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("すべての処理が完了しました。")
print("="*60)
