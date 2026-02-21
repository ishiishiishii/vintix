"""
ANYmalCの初期関節角度で立たせたときの実際の高さを測定するスクリプト
train.pyと同じ設定を使用

使い方:
  python check_anymalc_height.py           # ビューア表示
  python check_anymalc_height.py --no-viewer  # 結果のみ表示して終了（CI/Docker用）
"""
import genesis as gs
import torch
import numpy as np
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--no-viewer", action="store_true", help="ビューアを開かず結果のみ表示")
args = parser.parse_args()

# 初期化
gs.init()

# URDFパス（Docker/ローカル両対応）
_script_dir = Path(__file__).resolve().parent
_genesis_root = _script_dir.parent.parent
_urdf_path = _genesis_root / "genesis" / "assets" / "urdf" / "anymal_c" / "urdf" / "anymal_c.urdf"
if not _urdf_path.exists():
    _urdf_path = Path("/workspace/Genesis/genesis/assets/urdf/anymal_c/urdf/anymal_c.urdf")

# train.pyと同じデフォルト関節角度（get_anymalc_cfgs()から）
default_joint_angles = {
    # 左前脚 (LF - Left Front)
    "LF_HAA": -0.3,     # Hip Abduction/Adduction
    "LF_HFE": 0.8,      # Hip Flexion/Extension
    "LF_KFE": -1.3,     # Knee Flexion/Extension
    # 右前脚 (RF - Right Front)
    "RF_HAA": 0.3,      # Hip Abduction/Adduction
    "RF_HFE": 0.8,      # Hip Flexion/Extension
    "RF_KFE": -1.3,     # Knee Flexion/Extension
    # 左後脚 (LH - Left Hind)
    "LH_HAA": -0.3,     # Hip Abduction/Adduction
    "LH_HFE": -0.8,     # Hip Flexion/Extension
    "LH_KFE": 1.3,      # Knee Flexion/Extension
    # 右後脚 (RH - Right Hind)
    "RH_HAA": 0.3,      # Hip Abduction/Adduction
    "RH_HFE": -0.8,     # Hip Flexion/Extension
    "RH_KFE": 1.3,      # Knee Flexion/Extension
}

# train.pyと同じジョイント名の順序（get_anymalc_cfgs()から）
joint_names = [
    "RH_HAA",
    "LH_HAA",
    "RF_HAA",
    "LF_HAA",
    "RH_HFE",
    "LH_HFE",
    "RF_HFE",
    "LF_HFE",
    "RH_KFE",
    "LH_KFE",
    "RF_KFE",
    "LF_KFE",
]

# train.pyと同じ初期姿勢（get_anymalc_cfgs()から）
base_init_pos = [0.0, 0.0, 0.55]  # 初期高さ（train.pyと同じ）
base_init_quat = [1.0, 0.0, 0.0, 0.0]  # train.pyと同じ

# train.pyと同じ目標高さ（get_anymalc_cfgs()から）
base_height_target = 0.3  # Go2と同じ（現在の設定）

# シーンを作成（ANYmalCの高さに合わせてカメラ位置を調整）
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
    viewer_options=gs.options.ViewerOptions(
        max_FPS=60,
        camera_pos=(3.0, 0.0, 2.5),  # カメラを高めに設定（ANYmalCは高さがあるため）
        camera_lookat=(0.0, 0.0, 0.55),  # ロボットの中心を見る
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
    rigid_options=gs.options.RigidOptions(
        dt=0.02,
        constraint_solver=gs.constraint_solver.Newton,
        enable_collision=True,
        enable_joint_limit=True,
    ),
    show_viewer=not args.no_viewer,
)

# 地面を追加
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# ロボットを追加（train.pyと同じbase_init_posに配置）
robot = scene.add_entity(
    gs.morphs.URDF(
        file=str(_urdf_path),
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

# 関節角度を設定した直後のベース高さを測定
base_pos_after_joints = robot.get_pos()
height_after_joints = base_pos_after_joints[0, 2].item()

# 足の高さを測定し、初期関節角度でのスタンディング高さを算出
# スタンディング高さ = 足が地面(z=0)に着いたときのベース高さ
# GenesisのANYmalCは fixed joint で FOOT が統合されるため、FOOT がなければ SHANK（すね先端）で代用
foot_heights = []
for link in robot.links:
    name_upper = link.name.upper()
    if "FOOT" in name_upper:
        try:
            pos = link.get_pos()
            z = pos[0, 2].item() if pos.dim() > 1 else pos[2].item()
            foot_heights.append(z)
        except Exception:
            pass
if not foot_heights:
    for link in robot.links:
        if "SHANK" in link.name.upper():
            try:
                pos = link.get_pos()
                z = pos[0, 2].item() if pos.dim() > 1 else pos[2].item()
                foot_heights.append(z)
            except Exception:
                pass
min_foot_height = min(foot_heights) if foot_heights else 0.0
# スタンディング高さ = base_z - min_foot_height（足を地面に着けるためにベースを下げた高さ）
standing_base_height = base_init_pos[2] - min_foot_height

# 測定結果を表示
print("\n" + "="*60)
print("ANYmalC 測定結果（物理シミュレーション実行前）")
print("="*60)
print(f"関節角度設定後のベース高さ: {height_after_joints:.4f}m ({height_after_joints*100:.2f}cm)")
if foot_heights:
    print(f"最低の足の高さ（SHANK中心）: {min_foot_height:.4f}m ({min_foot_height*100:.2f}cm)")
    print(f"  ※GenesisではFOOTがSHANKに統合されているため、実足先はこれより低い")
    print(f"")
    print(f"【初期関節角度でのスタンディング高さ（足が地面に着いたときのベース高さ）】")
    print(f"  {standing_base_height:.4f}m ({standing_base_height*100:.2f}cm)")
    print(f"  ※SHANK中心を地面とした値。実機の足先着地時はやや高め（目安+5〜10cm）")
print(f"")
print(f"現在の設定 (train.pyから):")
print(f"  base_init_pos: {base_init_pos[2]:.2f}m ({base_init_pos[2]*100:.2f}cm)")
print(f"  base_height_target: {base_height_target:.2f}m ({base_height_target*100:.2f}cm)")
print(f"")
print(f"測定値と設定値の比較:")
print(f"  初期関節角度でのスタンディング高さ: {standing_base_height*100:.2f}cm")
print(f"  base_init_pos: {base_init_pos[2]*100:.2f}cm (差: {abs(standing_base_height*100 - base_init_pos[2]*100):.2f}cm)")
print(f"  base_height_target: {base_height_target*100:.2f}cm (差: {abs(standing_base_height*100 - base_height_target*100):.2f}cm)")
print("="*60)
sys.stdout.flush()

if not args.no_viewer:
    print("\nビューアで確認できます。Ctrl+Cで終了します。")
    print("（物理シミュレーションは実行していません）")
    sys.stdout.flush()
    try:
        import time
        if hasattr(scene, "_visualizer") and scene._visualizer is not None:
            scene._visualizer.update(force=True, auto=True)
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n終了します。")

