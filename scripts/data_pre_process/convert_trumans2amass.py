import numpy as np
import joblib
import sys
import os
from scipy.spatial.transform import Rotation as sRot

def convert_smplx_to_amass(smplx_pkl_path, output_npz_path):
    """
    Convert SMPL-X results pkl file to AMASS npz format
    """
    try:
        # Load SMPL-X data
        smplx_data = joblib.load(smplx_pkl_path)
        
        print(f"Loaded SMPL-X data with keys: {list(smplx_data.keys())}")
        
        # Extract required fields
        global_orient = smplx_data.get('global_orient', None)  # Root orientation
        body_pose = smplx_data.get('body_pose', None)  # Body joint poses
        transl = smplx_data.get('transl', None)  # Translation
        betas = smplx_data.get('betas', None)  # Shape parameters
        gender = smplx_data.get('gender', 'neutral')  # Gender
        
        # Validate required fields
        if global_orient is None or body_pose is None or transl is None:
            raise ValueError("Missing required pose or translation data")
        
        # Convert numpy arrays if they're torch tensors
        if hasattr(global_orient, 'numpy'):
            global_orient = global_orient.detach().cpu().numpy()
        if hasattr(body_pose, 'numpy'):
            body_pose = body_pose.detach().cpu().numpy()
        if hasattr(transl, 'numpy'):
            transl = transl.detach().cpu().numpy()
        if hasattr(betas, 'numpy'):
            betas = betas.detach().cpu().numpy()
        
        # Ensure correct shapes
        if global_orient.ndim == 1:
            global_orient = global_orient.reshape(1, -1)
        if body_pose.ndim == 1:
            body_pose = body_pose.reshape(1, -1)
        if transl.ndim == 1:
            transl = transl.reshape(1, -1)
        
        # Get number of frames
        N = max(global_orient.shape[0], body_pose.shape[0], transl.shape[0])
        
        # Repeat data if single frame
        if global_orient.shape[0] == 1 and N > 1:
            global_orient = np.repeat(global_orient, N, axis=0)
        if body_pose.shape[0] == 1 and N > 1:
            body_pose = np.repeat(body_pose, N, axis=0)
        if transl.shape[0] == 1 and N > 1:
            transl = np.repeat(transl, N, axis=0)
        
        # 定义坐标系转换矩阵：相机坐标系转世界坐标系（Z轴朝上）
        R_cam_to_world = np.array([[1, 0, 0],
                                   [0, 0, -1],
                                   [0, 1, 0]])
        
        # 转换平移到世界坐标系
        trans_world = (R_cam_to_world @ transl.T).T
        
        # 定义根关节校正旋转（只需X轴旋转-90度）
        R_correct = sRot.from_euler('x', 90, degrees=True).as_matrix()

        
        # 处理根关节旋转
        corrected_global_orient = np.zeros_like(global_orient)
        for i in range(N):
            # 将轴角转换为旋转矩阵
            R_root = sRot.from_rotvec(global_orient[i]).as_matrix()
            # 应用校正
            R_root_corrected = R_correct @ R_root
            # 转换回轴角
            corrected_global_orient[i] = sRot.from_matrix(R_root_corrected).as_rotvec()
        
        # Construct poses array: [corrected_global_orient (3) + body_pose (63) = 66 dims]
        # SMPL-X has more joints than SMPL, so we take first 63 dims of body_pose
        if body_pose.shape[1] > 63:
            body_pose = body_pose[:, :63]
        elif body_pose.shape[1] < 63:
            # Pad with zeros if less than 63
            padding = np.zeros((body_pose.shape[0], 63 - body_pose.shape[1]))
            body_pose = np.concatenate([body_pose, padding], axis=1)
        
        poses = np.concatenate([corrected_global_orient, body_pose], axis=1)
        
        # Handle betas (shape parameters)
        if betas is None:
            betas = np.zeros(10)  # Default SMPL shape parameters
        elif betas.ndim == 2:
            betas = betas.mean(axis=0)  # Use mean if multiple frames
        elif betas.ndim == 1:
            betas = betas  # Keep as is if already 1D
        
        # Handle gender
        if isinstance(gender, (list, np.ndarray)):
            gender = gender[0] if len(gender) > 0 else 'neutral'
        if isinstance(gender, bytes):
            gender = gender.decode('utf-8')
        
        # Create AMASS-compatible data structure
        amass_data = {
            'poses': poses.astype(np.float32),
            'trans': trans_world.astype(np.float32),  # 使用转换后的世界坐标
            'betas': betas.astype(np.float32),
            'gender': gender,
            'mocap_framerate': 30.0  # Default framerate
        }
        
        print(f"Converted data shapes:")
        print(f"  poses: {amass_data['poses'].shape}")
        print(f"  trans: {amass_data['trans'].shape}")
        print(f"  betas: {amass_data['betas'].shape}")
        print(f"  gender: {amass_data['gender']}")
        print(f"  framerate: {amass_data['mocap_framerate']}")
        
        # Save as npz file
        np.savez(output_npz_path, **amass_data)
        print(f"Successfully saved converted data to: {output_npz_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_trumans2amass.py <input_pkl_path> <output_npz_path>")
        print("Example: python convert_trumans2amass.py data/amass/2023-01-17@00-46-07_smplx_results.pkl data/amass/converted_motion.npz")
        sys.exit(1)
    
    input_pkl = sys.argv[1]
    output_npz = sys.argv[2]
    
    if not os.path.exists(input_pkl):
        print(f"Error: Input file '{input_pkl}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_npz)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success = convert_smplx_to_amass(input_pkl, output_npz)
    
    if success:
        print("\nConversion completed successfully!")
        print(f"You can now use the converted file with load_amass_data:")
        print(f"load_amass_data('{output_npz}')")
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()