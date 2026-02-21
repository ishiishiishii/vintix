import cv2
import genesis as gs

# Genesis初期化（CPUでOK）
gs.init(backend=gs.cpu)

# シーン作成（Viewer付き）
scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
)

# 簡単なオブジェクトを追加（地面）
scene.add_entity(gs.morphs.Plane())

# GUI付きカメラを追加（OpenCVウィンドウ用）
cam = scene.add_camera(
    res=(1280, 960),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=True  # これがOpenCVウィンドウを表示する
)

# シーン構築
scene.build()

# メインループ（表示＋レンダリング）
for i in range(300):
    scene.step()  # シーンを1ステップ進める

    # 画像をレンダリング（4つの出力）
    rgb, depth, segmentation, normal = cam.render(
        depth=True, segmentation=True, normal=True
    )

    # OpenCVでウィンドウ更新を保証（真っ黒対策）
    cv2.waitKey(1)
