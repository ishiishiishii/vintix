import cv2
import genesis as gs
gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # `world`の原点座標系を可視化
        world_frame_size = 1.0, # 座標系の長さ（メートル単位）
        show_link_frame  = False, # エンティティリンクの座標系は非表示
        show_cameras     = False, # 追加されたカメラのメッシュと視錐台を非表示
        plane_reflection = True, # 平面反射を有効化
        ambient_light    = (0.1, 0.1, 0.1), # 環境光設定
    ),
    renderer = gs.renderers.Rasterizer(), # カメラレンダリングにラスタライザを使用
)


plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),)

cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = True
)

scene.build()

# RGB、深度、セグメンテーションマスク、法線マップをレンダリング
for i in range(1000):
    scene.step()

    rgb, depth, segmentation, normal = cam.render(depth=True, segmentation=True, normal=True)
    cv2.waitKey(1)

