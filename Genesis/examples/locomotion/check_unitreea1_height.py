"""
UnitreeA1の初期関節角度で立たせたときの実際の高さを測定するスクリプト
train.pyと同じ設定を使用
関節角度はそのままで、足が地面に届くようにbase_init_posを調整する
"""
import genesis as gs
import torch
import numpy as np
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

# train.pyと同じ初期姿勢（後で調整される）
base_init_pos = [0.0, 0.0, 0.30]  # 初期値（後で調整）
base_init_quat = [1.0, 0.0, 0.0, 0.0]  # train.pyと同じ

# train.pyと同じ目標高さ（get_unitreea1_cfgs()から）
base_height_target = 0.27  # 目標高さ0.27m（27cm）

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
    show_viewer=True,  # ビューアありで実行
)

# 地面を追加
scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

# ロボットを追加（最初はtrain.pyと同じbase_init_posに配置）
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/workspace/Genesis/genesis/assets/urdf/unitree_a1/urdf/a1.urdf",
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

# 物理シミュレーションを実行せずに、関節角度を設定した直後のベース高さと足の高さを測定
base_pos = robot.get_pos()
initial_base_height = base_pos[0, 2].item()

# 利用可能なリンク名を確認（デバッグ用）
print("\n利用可能なリンク名を確認中...")
available_links = [link.name for link in robot.links]
foot_related_links = [name for name in available_links if 'foot' in name.lower() or 'calf' in name.lower()]
print(f"足関連のリンク: {foot_related_links}")
sys.stdout.flush()

# 足のリンクの位置を取得（FR_foot, FL_foot, RR_foot, RL_foot）
# もしfootリンクが見つからない場合は、calfリンクの位置から推定
foot_link_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
calf_link_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
foot_heights = []

# まずfootリンクを試す
for foot_name in foot_link_names:
    try:
        foot_link = robot.get_link(foot_name)
        foot_pos = foot_link.get_pos()
        foot_height = foot_pos[0, 2].item()
        foot_heights.append(foot_height)
    except Exception as e:
        # footリンクが見つからない場合は、calfリンクから推定（calfの下端は約-0.213m下）
        try:
            calf_name = foot_name.replace("_foot", "_calf")
            calf_link = robot.get_link(calf_name)
            calf_pos = calf_link.get_pos()
            # calfリンクの下端は約-0.213m下（URDFから確認、A1も同様の構造と仮定）
            estimated_foot_height = calf_pos[0, 2].item() - 0.213
            foot_heights.append(estimated_foot_height)
            print(f"  {foot_name}が見つからないため、{calf_name}から推定: {estimated_foot_height*100:.2f}cm")
        except Exception as e2:
            print(f"Warning: Could not get position for {foot_name} or {calf_name}: {e2}")
            pass

# 最低の足の高さを取得
if foot_heights:
    min_foot_height = min(foot_heights)
    max_foot_height = max(foot_heights)
else:
    # 足のリンクが見つからない場合は、ベース高さから推定
    min_foot_height = initial_base_height - 0.3  # 仮の値
    max_foot_height = initial_base_height - 0.2  # 仮の値

# 足が地面に届くようにbase_init_posを調整
# 最低の足の高さが0になるようにbase_init_posを下げる
ground_level = 0.0
required_adjustment = min_foot_height - ground_level
adjusted_base_init_pos = base_init_pos[2] - required_adjustment

print("\n" + "="*60)
print("測定結果（物理シミュレーション実行前）")
print("="*60)
print(f"初期関節角度設定時のベース高さ: {initial_base_height:.4f}m ({initial_base_height*100:.2f}cm)")
if foot_heights:
    print(f"最低の足の高さ: {min_foot_height:.4f}m ({min_foot_height*100:.2f}cm)")
    print(f"最高の足の高さ: {max_foot_height:.4f}m ({max_foot_height*100:.2f}cm)")
print(f"")
print(f"現在の設定 (train.pyから):")
print(f"  base_init_pos: {base_init_pos[2]:.2f}m ({base_init_pos[2]*100:.2f}cm)")
print(f"  base_height_target: {base_height_target:.2f}m ({base_height_target*100:.2f}cm)")
print(f"")
print(f"調整推奨値:")
print(f"  現在の最低足高さ: {min_foot_height*100:.2f}cm")
print(f"  地面レベル: 0.00cm")
print(f"  必要な調整: {required_adjustment*100:.2f}cm")
print(f"  推奨base_init_pos: {adjusted_base_init_pos:.4f}m ({adjusted_base_init_pos*100:.2f}cm)")
print(f"")
print(f"測定値と設定値の比較:")
print(f"  ベース高さ測定値: {initial_base_height*100:.2f}cm")
print(f"  base_init_pos: {base_init_pos[2]*100:.2f}cm (差: {abs(initial_base_height*100 - base_init_pos[2]*100):.2f}cm)")
print(f"  base_height_target: {base_height_target*100:.2f}cm (差: {abs(initial_base_height*100 - base_height_target*100):.2f}cm)")
print("="*60)
print("\nビューアで確認できます。Ctrl+Cで終了します。")
print("（物理シミュレーションは実行していません）")
sys.stdout.flush()  # 出力を確実にフラッシュ

# 調整されたbase_init_posでロボットを再配置
if required_adjustment > 0:
    print(f"\n足が地面に届くようにbase_init_posを{adjusted_base_init_pos*100:.2f}cmに調整します...")
    sys.stdout.flush()
    
    # ロボットの位置を調整
    robot.set_pos(
        pos=torch.tensor([[0.0, 0.0, adjusted_base_init_pos]], device=gs.device, dtype=gs.tc_float),
        envs_idx=[0],
    )
    
    # 関節角度を再設定（位置変更後も維持）
    robot.set_dofs_position(
        position=default_dof_pos.unsqueeze(0),
        dofs_idx_local=motors_dof_idx,
        zero_velocity=True,
        envs_idx=[0],
    )
    
    # 再測定
    base_pos_after = robot.get_pos()
    initial_base_height_after = base_pos_after[0, 2].item()
    
    # 足の高さを再測定
    foot_heights_after = []
    for foot_name in foot_link_names:
        try:
            foot_link = robot.get_link(foot_name)
            foot_pos = foot_link.get_pos()
            foot_height = foot_pos[0, 2].item()
            foot_heights_after.append(foot_height)
        except Exception as e:
            # footリンクが見つからない場合は、calfリンクから推定
            try:
                calf_name = foot_name.replace("_foot", "_calf")
                calf_link = robot.get_link(calf_name)
                calf_pos = calf_link.get_pos()
                estimated_foot_height = calf_pos[0, 2].item() - 0.213
                foot_heights_after.append(estimated_foot_height)
            except Exception as e2:
                print(f"Warning: Could not get position for {foot_name} or {calf_name} after adjustment: {e2}")
                pass
    
    if foot_heights_after:
        min_foot_height_after = min(foot_heights_after)
        max_foot_height_after = max(foot_heights_after)
        print(f"\n調整後の結果:")
        print(f"  最低の足の高さ: {min_foot_height_after:.4f}m ({min_foot_height_after*100:.2f}cm)")
        print(f"  最高の足の高さ: {max_foot_height_after:.4f}m ({max_foot_height_after*100:.2f}cm)")
        print(f"  ベース高さ: {initial_base_height_after:.4f}m ({initial_base_height_after*100:.2f}cm)")
        print(f"\n推奨されるtrain.pyの設定:")
        print(f"  base_init_pos: [0.0, 0.0, {adjusted_base_init_pos:.3f}]  # {adjusted_base_init_pos*100:.1f}cm")
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
