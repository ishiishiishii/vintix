#!/usr/bin/env python3
"""
Laikago URDFビューアー（環境光を追加）
"""
import genesis as gs

def view_laikago_with_light(urdf_file="urdf/laikago/urdf/laikago_toes_zup.urdf"):
    gs.init(backend=gs.cpu)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(4.0, 2.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.2),
            camera_fov=50,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=0.3,
            show_link_frame=False,
            ambient_light=(0.5, 0.5, 0.5),  # 環境光を明るく設定
        ),
        show_viewer=True,
    )
    
    # URDFを読み込む
    entity = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_file,
            collision=False,
            scale=1.0,
        ),
        surface=gs.surfaces.Default(
            vis_mode="visual",
        ),
    )
    
    scene.build(compile_kernels=False)
    
    print(f"URDFファイルを読み込みました: {urdf_file}")
    print("ビューアーを起動しました。")
    print("マウスでカメラを操作できます。")
    
    # メインループ
    while scene.viewer.is_alive():
        scene.visualizer.update(force=True)

if __name__ == "__main__":
    import sys
    urdf_file = sys.argv[1] if len(sys.argv) > 1 else "urdf/laikago/urdf/laikago_toes_zup.urdf"
    view_laikago_with_light(urdf_file)

