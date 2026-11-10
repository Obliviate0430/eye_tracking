import cv2
import numpy as np

cap = None
camera_id_found = None

# 1. 搜索可用摄像头
for camera_id in range(5):

    test_cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not test_cap.isOpened():
        print(f"Camera {camera_id}: Cannot open")
        test_cap.release()
        continue

    ret, frame = test_cap.read()

    if ret and frame is not None:
        print(f"Camera {camera_id}: Connected")
        print("Resolution:", frame.shape)
        print("Mean brightness:", np.mean(frame))
        print("Min/Max:", frame.min(), frame.max())

        cap = test_cap
        camera_id_found = camera_id
        break

    test_cap.release()

if cap is None:
    raise RuntimeError("No available camera found!")

print(f"\nUsing Camera {camera_id_found}")

# 2. 只创建一个显示窗口
cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to read frame")
        break

    frame_count += 1

    # 每隔 60 帧输出一次亮度
    if frame_count % 60 == 0:
        print(
            f"Frame {frame_count} | "
            f"Brightness: {np.mean(frame):.2f} | "
            f"Min/Max: {frame.min()}/{frame.max()}"
        )

    cv2.imshow("Camera Preview", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()