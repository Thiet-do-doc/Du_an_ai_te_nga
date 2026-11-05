
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR / "data" / "processed"
    / "pose_features" / "train"
)

KEYPOINT_NAMES = [
    "Mũi",
    "Mắt trái",
    "Mắt phải",
    "Tai trái",
    "Tai phải",
    "Vai trái",
    "Vai phải",
    "Khuỷu tay trái",
    "Khuỷu tay phải",
    "Cổ tay trái",
    "Cổ tay phải",
    "Hông trái",
    "Hông phải",
    "Đầu gối trái",
    "Đầu gối phải",
    "Mắt cá chân trái",
    "Mắt cá chân phải",
]

files = sorted(DATA_DIR.glob("*.npy"))

if not files:
    raise FileNotFoundError(
        f"Không tìm thấy file .npy tại: {DATA_DIR}"
    )

# Đọc file đầu tiên
file_path = files[0]
data = np.load(file_path, allow_pickle=False)

print("Tên file:", file_path.name)
print("Kích thước dữ liệu:", data.shape)
print("Kiểu dữ liệu:", data.dtype)

# Chọn frame đầu tiên
frame_index = 0
frame = data[frame_index].reshape(17, 3)

df = pd.DataFrame(
    frame,
    columns=["x", "y", "confidence"]
)

df.insert(0, "Keypoint", KEYPOINT_NAMES)

print(f"\n17 keypoint tại frame {frame_index + 1}:")
print(df.to_string(index=False))
