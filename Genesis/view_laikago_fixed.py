#!/usr/bin/env python3
"""
Laikago URDFビューアー（カメラ位置とライティングを調整）
"""
import genesis as gs
import numpy as np

def view_laikago(urdf_file="urdf/laikago/urdf/laikago_toes_zup.urdf"):
    gs.init(backend=gs.cpu)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(4.0, 2.0, 1.5),  # カメラ位置を調整
            camera_lookat=(0.0, 0.0, 0.2),  # ロボットの中心を見る
            camera_fov=50,  # 視野角を広げる
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,  # ワールド座標系を表示
            world_frame_size=0.5,
            show_link_frame=False,
            ambient_light=(0.3, 0.3, 0.3),  # 環境光を明るく設定
        ),
        show_viewer=True,
    )
    
    # 地面を追加（視認性向上のため）
    plane = scene.add_entity(
        gs.morphs.Plane(
            pos=(0.0, 0.0, -0.1),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.5, 0.5),
        ),
    )
    
    # URDFを読み込む
    entity = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_file,
            collision=False,
            scale=1.0,
            pos=(0.0, 0.0, 0.0),  # 原点に配置
        ),
        surface=gs.surfaces.Default(
            vis_mode="visual",
            color=(0.8, 0.8, 0.8),  # 明るい色を設定
        ),
    )
    
    scene.build(compile_kernels=False)
    
    print(f"URDFファイルを読み込みました: {urdf_file}")
    print("ビューアーを起動しました。ウィンドウを閉じると終了します。")
    print("カメラ位置:", scene.viewer.camera_pose)
    
    # メインループ
    while scene.viewer.is_alive():
        scene.visualizer.update(force=True)
    
    print("ビューアーを終了しました。")

if __name__ == "__main__":
    import sys
    urdf_file = sys.argv[1] if len(sys.argv) > 1 else "urdf/laikago/urdf/laikago_toes_zup.urdf"
    view_laikago(urdf_file)

