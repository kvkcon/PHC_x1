import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch.multiprocessing as mp
from smpl_sim.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser, 
)

import joblib
import torch
import torch.nn.functional as F
import math
from smpl_sim.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from tqdm import tqdm
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from phc.torch_humanoid_batch import Humanoid_Batch
from smpl_sim.utils.smoothing_utils import gaussian_kernel_1d, gaussian_filter_1d_batch
from easydict import EasyDict
import hydra
from omegaconf import DictConfig, OmegaConf

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return 
    framerate = entry_data['mocap_framerate']

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
    
def process_motion(key_names, key_name_to_pkls, cfg):
    device = torch.device("cpu")
    
    humanoid_fk = Humanoid_Batch(cfg.robot) # load forward kinematics model
    num_augment_joint = len(cfg.robot.extend_config)

    #### Define correspondences between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment 
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    body_link_len = len(robot_joint_names_augment)
    
    # Parse DOF_WEIGHT from config
    dof_weights = torch.ones(humanoid_fk.num_dof)
    if hasattr(cfg.robot, 'DOF_WEIGHT') and cfg.robot.DOF_WEIGHT is not None:
        default_weight = cfg.robot.DOF_WEIGHT.get('default', 1.0)
        dof_weights.fill_(default_weight)
        for idx, weight in cfg.robot.DOF_WEIGHT.items():
            if idx != 'default' and isinstance(idx, int):
                if 0 <= idx < humanoid_fk.num_dof:
                    dof_weights[idx] = weight
    dof_weights = dof_weights.to(device)
    
    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    shape_new, scale = joblib.load(f"data/{cfg.robot.humanoid_type}/shape_optimized_v1.pkl") # TODO: run fit_smple_shape to get this
    
    
    all_data = {}
    pbar = tqdm(key_names, position=0, leave=True)
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None: continue
        skip = int(amass_data['fps']//30)
        trans = torch.from_numpy(amass_data['trans'][::skip])
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(amass_data['pose_aa'][::skip]).float()
        
        if N < 10:
            print("to short")
            continue

        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] -= verts[0, :, 2].min().item()
        
        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        gt_root_rot_quat = torch.from_numpy((sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float() # can't directly use this 
        gt_root_rot = torch.from_numpy(sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()).float() # so only use the heading. 
        
        # Initialize dof_pos for actuated joints
        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

        dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
        root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
        root_pos_offset = Variable(torch.zeros(1, 3), requires_grad=True)
        optimizer = torch.optim.Adam([dof_pos_new, root_rot_new, root_pos_offset], lr=0.02)

        kernel_size = 5  # Size of the Gaussian kernel
        sigma = 0.75  # Standard deviation of the Gaussian kernel
        B, T, J, D = dof_pos_new.shape    

        # Define indices for actuated joints within robot_joint_names_augment
        # actuated_joint_names = ['left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'waist_pitch_link', 'head_yaw_link', 'head_pitch_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link', 'left_wrist_pitch_link', 'left_hand_yaw_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_yaw_link', 'right_wrist_pitch_link', 'right_hand_yaw_link']
        actuated_joint_indices = humanoid_fk.actuated_joints_idx #[robot_joint_names_augment.index(name) for name in actuated_joint_names]
        
        for iteration in range(cfg.get("fitting_iterations", 300)):
            # print(f"iteration: {iteration}")
            # print(f"root_pos_offset: {root_pos_offset}")
            # print(f"root_rot_new.shape: {root_rot_new.shape}")
            # print(f"humanoid_fk.dof_axis.shape: {humanoid_fk.dof_axis.shape}")
            # print(f"dof_pos_new.shape: {dof_pos_new.shape}")
            # print(f"dof_pos_new: {dof_pos_new[0,94,:,0]}")
            # print(f"N: {N}")
            # print(f"num_augment_joint: {num_augment_joint}")
            
            # Initialize pose_aa_h1_new with zeros for all 42 joints
            pose_aa_h1_new = torch.zeros((1, N, body_link_len, 3)).to(device)
            
            # Set root rotation (first joint, index 0)
            pose_aa_h1_new[:, :, 0, :] = root_rot_new[None, :, :]
            
            # Map actuated joints' DoF to corresponding indices
            for i, idx in enumerate(actuated_joint_indices):
                pose_aa_h1_new[:, :, idx, :] = humanoid_fk.dof_axis[i] * dof_pos_new[:, :, i, 0][:, :, None]
            
            # print(f"pose_aa_h1_new.shape: {pose_aa_h1_new.shape}")
            # print(f"root_trans_offset.shape: {root_trans_offset.shape}")                        
            # print(f"root_pos_offset.shape: {root_pos_offset.shape}")
            fk_return = humanoid_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ] + root_pos_offset )
            
            if num_augment_joint > 0:
                diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            else:
                diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
                
            loss_g = diff.norm(dim = -1).mean() + 0.05 * torch.mean(torch.square(dof_pos_new) * dof_weights[None, None, :, None])
            loss = loss_g
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

            pbar.set_description_str(f"{data_key}-Iter: {iteration} \t {loss.item() * 1000:.3f}")
            dof_pos_new.data = gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None, ], kernel_size, sigma).transpose(2, 1)[..., None]
            
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt
        
        j3d = fk_return.global_translation[0, :, :, :].detach().numpy()
        # print(f"j3d.shape: {j3d.shape}")
        j3d_joints = joints.detach().numpy()
        idx=94

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(90, 0)
        ax.scatter(j3d[idx, :,0], j3d[idx, :,1], j3d[idx, :,2])
        ax.scatter(j3d_joints[idx, :,0], j3d_joints[idx, :,1], j3d_joints[idx, :,2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        drange = 2
        ax.set_xlim(0, 3.5)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(0, drange)
        plt.show()
            
        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
        
        # Recompute pose_aa_h1_new for final output
        pose_aa_h1_new = torch.zeros((1, N, body_link_len, 3)).to(device)
        pose_aa_h1_new[:, :, 0, :] = root_rot_new[None, :, :]
        for i, idx in enumerate(actuated_joint_indices):
            pose_aa_h1_new[:, :, idx, :] = humanoid_fk.dof_axis[i] * dof_pos_new[:, :, i, 0][:, :, None]

        root_trans_offset_dump = (root_trans_offset + root_pos_offset ).clone()

        # Move to ground
        combined_mesh = humanoid_fk.mesh_fk(pose_aa_h1_new[:, :1].detach(), root_trans_offset_dump[None, :1].detach())
        height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
        
        root_trans_offset_dump[..., 2] -= height_diff
        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff
        
        data_dump = {
                    "root_trans_offset": root_trans_offset_dump.squeeze().detach().numpy(),
                    "pose_aa": pose_aa_h1_new.squeeze().detach().numpy(),   
                    "dof": dof_pos_new.squeeze().detach().numpy(), 
                    "dof_index": np.array(humanoid_fk.actuated_joints_idx),
                    "dof_names": [humanoid_fk.body_names[i] for i in humanoid_fk.actuated_joints_idx],
                    "dof_axis": humanoid_fk.dof_axis.detach().numpy(),
                    "root_rot": sRot.from_rotvec(root_rot_new.detach().numpy()).as_quat(),
                    "smpl_joints": joints_dump, 
                    "fps": 30
                    }
        all_data[data_key] = data_dump
    return all_data
        

@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    if "amass_root" in cfg:
        amass_root = cfg.amass_root
    else:
        raise ValueError("amass_root is not specified in the config")
    
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    key_names = ["0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", "") for data_path in all_pkls]
    if not cfg.get("fit_all", False):
        key_names = [f"{cfg.motion_name}"]
    torch.set_num_threads(1)
    mp.set_sharing_strategy('file_descriptor')
    jobs = key_names
    num_jobs = 30
    chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i], key_name_to_pkls, cfg) for i in range(len(jobs))]
    if len(job_args) == 1:
        all_data = process_motion(key_names, key_name_to_pkls, cfg)
    else:
        try:
            pool = mp.Pool(num_jobs)   # multi-processing
            all_data_list = pool.starmap(process_motion, job_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        all_data = {}
        for data_dict in all_data_list:
            all_data.update(data_dict)
    if len(all_data) == 1:
        data_key = list(all_data.keys())[0]
        os.makedirs(f"data/{cfg.robot.humanoid_type}/v1/singles", exist_ok=True)
        dumped_file = f"data/{cfg.robot.humanoid_type}/v1/singles/{data_key}.pkl"
        print(dumped_file)
        joblib.dump(all_data, dumped_file)
    else:
        os.makedirs(f"data/{cfg.robot.humanoid_type}/v1/", exist_ok=True)
        joblib.dump(all_data, f"data/{cfg.robot.humanoid_type}/v1/amass_all.pkl")
    


if __name__ == "__main__":
    main()