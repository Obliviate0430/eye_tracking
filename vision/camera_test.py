
import cv2
import numpy as np

for camera_id in range(5):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Camera {camera_id}: Cannot open")
        cap.release()
        continue

    ret, frame = cap.read()

    if ret and frame is not None:
        print(f"\nCamera {camera_id}:")
        print("Resolution:", frame.shape)
        print("Mean brightness:", np.mean(frame))
        print("Min/Max:", frame.min(), frame.max())

        cv2.imshow(f"Camera {camera_id}", frame)
        cv2.waitKey(1500)

    cap.release()

cv2.destroyAllWindows()
