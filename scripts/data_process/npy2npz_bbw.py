import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as sRot

# 定义坐标系转换矩阵
R_cam_to_world = np.array([[1, 0, 0],
                           [0, 0, 1],
                           [0, -1, 0]])

# 检查命令行参数
if len(sys.argv) != 2:
    print("Usage: python npy2npz_convert.py <video_name>")
    sys.exit(1)

video_name = sys.argv[1]
tram_file_path = f"../tram/results/{video_name}/hps/hps_track_0.npy"

# 检查文件是否存在
if not os.path.exists(tram_file_path):
    print(f"Error: File {tram_file_path} does not exist.")
    sys.exit(1)

# 加载数据
track_data = np.load(tram_file_path, allow_pickle=True).item()

# 检查必需的键
required_keys = ['pred_rotmat', 'pred_trans', 'pred_shape', 'pred_cam']
for key in required_keys:
    if key not in track_data:
        print(f"Error: Missing key '{key}' in {tram_file_path}")
        sys.exit(1)

pred_rotmat = track_data["pred_rotmat"].numpy()
print("原始根关节旋转矩阵 (Frame 0):\n", pred_rotmat[0, 0])
trans = track_data["pred_trans"].numpy().squeeze(1)  # [N, 3]
trans_world = (R_cam_to_world @ trans.T).T  # 转换为世界坐标系
shape = track_data["pred_shape"].numpy()
duration = track_data['pred_cam'].shape[0]

# 定义根关节校正旋转
R_correct = sRot.from_euler('z', 90, degrees=True).as_matrix()

# 转换姿态数据
poses = np.zeros((duration, 72))
for i in range(duration):
    for j in range(24):
        if j == 0:  # 根关节
            R_root_cam = pred_rotmat[i, 0]
            R_root_world = R_cam_to_world @ R_root_cam @ R_cam_to_world.T
            axis_angle = sRot.from_matrix(R_root_world).as_rotvec()
            poses[i, :3] = axis_angle
        else:  # 其他关节
            axis_angle = sRot.from_matrix(pred_rotmat[i, j]).as_rotvec()
            poses[i, 3 + 3*(j-1) : 3 + 3*(j-1) + 3] = axis_angle

# 处理平移和形状
N = duration
betas = shape.mean(axis=0)  # 使用 numpy 的 mean

# 填充缺失数据
gender = "neutral"
mocap_framerate = 30
dmpls = np.zeros((N, 8))

# 确保输出目录存在
output_dir = f"scripts/data_process/npz_transfered/{video_name}"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "24dof_transfered.npz")

# 保存数据
np.savez_compressed(
    output_path,
    trans=trans_world,
    gender=gender,
    mocap_framerate=mocap_framerate,
    betas=betas,
    dmpls=dmpls,
    poses=poses
)
print('npz file saved at:', output_path)