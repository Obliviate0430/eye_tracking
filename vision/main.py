import os
import shutil

source = r"C:\Users\wangg\Desktop\wgs\AAA博一\学习\sj_test\face_landmarker.task"

target_dir = r"C:\mediapipe_models"
target = os.path.join(target_dir, "face_landmarker.task")

os.makedirs(target_dir, exist_ok=True)

shutil.copy2(source, target)

print("Copied:", os.path.exists(target))
print("File size:", os.path.getsize(target))
print("Target:", target)