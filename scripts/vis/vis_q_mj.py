import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())


from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "T":
        print("next")
        motion_id += 1
        curr_motion_key = motion_data_keys[motion_id]
        print(curr_motion_key)
    else:
        print("not mapped", chr(keycode))
    
    
        
@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    device = torch.device("cpu")
    humanoid_xml = cfg.robot.asset.assetFileName
    sk_tree = SkeletonTree.from_mjcf(humanoid_xml)
    
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
    
    motion_file = f"data/{cfg.robot.humanoid_type}/v1/singles/{cfg.motion_name}.pkl"
    print(motion_file)
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    
    
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)

    RECORDING = False
    
    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(50):
            # add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.04, np.array([1, 0, 0, 1]))
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.02, np.array([1, 0, 0, 1]))
        # for _ in range(24):
        #     add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            step_start = time.time()
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            curr_time = int(time_step/dt) % curr_motion['dof'].shape[0]
            
                        # 添加打印信息来查看顺序
            if time_step == 0:  # 只在开始时打印一次
                print("\n=== Motion Data Info ===")
                print(f"curr_motion['dof'] shape: {curr_motion['dof'].shape}")
                print(f"curr_motion['dof'][0]: {curr_motion['dof'][76]}")
                
                # 添加motion file中的dof信息
                if 'dof_names' in curr_motion:
                    print(f"Motion dof_names: {curr_motion['dof_names']}")
                else:
                    print("Motion dof_names: Not available in motion file")
                    
                if 'dof_index' in curr_motion:
                    print(f"Motion dof_index: {curr_motion['dof_index']}")
                else:
                    print("Motion dof_index: Not available in motion file")
                    
                if 'dof_axis' in curr_motion:
                    print(f"Motion dof_axis shape: {curr_motion['dof_axis'].shape}")
                    print(f"Motion dof_axis: {curr_motion['dof_axis']}")
                else:
                    print("Motion dof_axis: Not available in motion file")
                
                print("\n=== MuJoCo Model Info ===")
                print(f"mj_model.nq (total qpos): {mj_model.nq}")
                print(f"mj_model.nv (total qvel): {mj_model.nv}")
                print(f"mj_model.njnt (number of joints): {mj_model.njnt}")
                
                print("\n=== Joint Names and Info ===")
                for i in range(mj_model.njnt):
                    joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    joint_type = mj_model.jnt_type[i]
                    qpos_addr = mj_model.jnt_qposadr[i]
                    dof_addr = mj_model.jnt_dofadr[i]
                    print(f"Joint {i}: {joint_name}, type: {joint_type}, qpos_addr: {qpos_addr}, dof_addr: {dof_addr}")
                
                print("\n=== DOF Names and Axis ===")
                for i in range(mj_model.nv):
                    # 找到对应的joint
                    joint_id = -1
                    for j in range(mj_model.njnt):
                        if mj_model.jnt_dofadr[j] <= i < mj_model.jnt_dofadr[j] + (3 if mj_model.jnt_type[j] == 0 else 1):
                            joint_id = j
                            break
                    
                    if joint_id >= 0:
                        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                        # 获取关节轴信息
                        if hasattr(mj_model, 'jnt_axis') and joint_id < len(mj_model.jnt_axis):
                            axis = mj_model.jnt_axis[joint_id]
                        else:
                            axis = "N/A"
                        print(f"DOF {i}: joint={joint_name}, axis={axis}")
                
                print(f"\nqpos layout:")
                print(f"qpos[0:3]: root translation")
                print(f"qpos[3:7]: root rotation (quaternion)")
                print(f"qpos[7:]: joint DOFs (length: {mj_model.nq - 7})")
                print(f"Motion dof length: {curr_motion['dof'].shape[1]}")
                
                # 添加更详细的关节类型信息
                print("\n=== Joint Type Details ===")
                for i in range(mj_model.njnt):
                    joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    joint_type = mj_model.jnt_type[i]
                    joint_type_name = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(joint_type, "unknown")
                    if hasattr(mj_model, 'jnt_axis') and i < len(mj_model.jnt_axis):
                        axis = mj_model.jnt_axis[i]
                        print(f"Joint {i}: {joint_name}, type: {joint_type_name}({joint_type}), axis: {axis}")
                    else:
                        print(f"Joint {i}: {joint_name}, type: {joint_type_name}({joint_type}), axis: N/A")
                
                print("\n=== DOF Mapping Comparison ===")
                if 'dof_names' in curr_motion and 'dof_index' in curr_motion:
                    motion_dof_names = curr_motion['dof_names']
                    motion_dof_index = curr_motion['dof_index']
                    print("Motion DOF order:")
                    for i, (name, idx) in enumerate(zip(motion_dof_names, motion_dof_index)):
                        print(f"  Motion DOF {i}: {name} (body_index: {idx})")
                    
                    print("MuJoCo DOF order (excluding root):")
                    for i in range(mj_model.nv):
                        if i >= 6:  # Skip root DOFs (freejoint has 6 DOFs)
                            joint_id = -1
                            for j in range(mj_model.njnt):
                                if mj_model.jnt_dofadr[j] <= i < mj_model.jnt_dofadr[j] + (3 if mj_model.jnt_type[j] == 0 else 1):
                                    joint_id = j
                                    break
                            if joint_id >= 0:
                                joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                                print(f"  MuJoCo DOF {i-6}: {joint_name}")
                
            mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
            # # print(f"len of curr_motion['dof'][curr_time]: {len(curr_motion['dof'][curr_time])}")
            mj_data.qpos[7:] = curr_motion['dof'][curr_time]
                
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            # pose_aa_h1_new = torch.cat([torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()[None, :, None]), H1_ROTATION_AXIS * dof_pos[None, ..., None], torch.zeros((1, 1, 2, 3))], axis = 2).float()
            # for i in range(rg_pos_t.shape[1]):
            #     if not i in [20, 21, 22]:
            #         continue
            #     viewer.user_scn.geoms[i].pos = rg_pos_t[0, i]
                # viewer.user_scn.geoms[i].pos = mj_data.xpos[:][i]
            
            joint_gt = motion_data[curr_motion_key]['smpl_joints']
            
            for i in range(joint_gt.shape[1]):
                viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
                
            # joints_opt = kps_dict['j3d_h1'].squeeze()#[h1_joint_pick_idx]
            # for i in range(len(joints_opt)):
            #     viewer.user_scn.geoms[i].pos = joints_opt[i]
                

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # if RECORDING and time_step > motion_len:
            #     curr_start += num_motions
            #     motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
            #     time_step = 0

if __name__ == "__main__":
    main()
