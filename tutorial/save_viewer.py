import cv2
import genesis as gs
import numpy as np

gs.init(backend=gs.cpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml')
)

#camは保存可能なカメラでcv2を使う
cam = scene.add_camera(GUI=True)
scene.build()

#録画開始
cam.start_recording()

for i in range(120):
    scene.step()

    #カメラの位置を動かすコード
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5)
    )
    cam.render()

#録画を停止してビデオを保存'filename'を指定しない場合、呼び出し元のファイル名を使用して名前をつける
cam.stop_recording(save_to_filename='video.mp4', fps=60)
