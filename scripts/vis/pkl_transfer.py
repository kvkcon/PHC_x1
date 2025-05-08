import os
import joblib
import numpy as np
import argparse

def remap_pkl_body_order(input_pkl_path, output_pkl_path=None):
    """
    Remaps the body order in the PKL file to match the simulator's body list order.
    
    Args:
        input_pkl_path: Path to the input PKL file
        output_pkl_path: Path to save the remapped PKL file (if None, will auto-generate)
    """
    print(f"Processing: {input_pkl_path}")
    
    # Load the PKL file
    all_data = joblib.load(input_pkl_path)
    
    # Define the source and target body names
    source_body_names = [
        'x1-body', 'link_lumbar_yaw', 'link_lumbar_roll', 'link_lumbar_pitch', 
        'link_left_shoulder_pitch', 'link_left_shoulder_roll', 'link_left_shoulder_yaw', 
        'link_left_elbow_pitch', 'link_left_elbow_yaw', 'link_left_wrist_pitch', 
        'link_left_wrist_roll', 'link_right_shoulder_pitch', 'link_right_shoulder_roll', 
        'link_right_shoulder_yaw', 'link_right_elbow_pitch', 'link_right_elbow_yaw', 
        'link_right_wrist_pitch', 'link_right_wrist_roll', 'link_left_hip_pitch', 
        'link_left_hip_roll', 'link_left_hip_yaw', 'link_left_knee_pitch', 
        'link_left_ankle_pitch', 'link_left_ankle_roll', 'link_right_hip_pitch', 
        'link_right_hip_roll', 'link_right_hip_yaw', 'link_right_knee_pitch', 
        'link_right_ankle_pitch', 'link_right_ankle_roll', 'head_link'
    ]
    
    target_body_names = [
        'base_link', 'link_left_hip_pitch', 'link_left_hip_roll', 'link_left_hip_yaw', 
        'link_left_knee_pitch', 'link_left_ankle_pitch', 'link_left_ankle_roll', 
        'link_lumbar_yaw', 'link_lumbar_roll', 'link_lumbar_pitch', 
        'link_left_shoulder_pitch', 'link_left_shoulder_roll', 'link_left_shoulder_yaw', 
        'link_left_elbow_pitch', 'link_left_elbow_yaw', 'link_left_wrist_pitch', 
        'link_left_wrist_roll', 'link_right_shoulder_pitch', 'link_right_shoulder_roll', 
        'link_right_shoulder_yaw', 'link_right_elbow_pitch', 'link_right_elbow_yaw', 
        'link_right_wrist_pitch', 'link_right_wrist_roll', 'link_right_hip_pitch', 
        'link_right_hip_roll', 'link_right_hip_yaw', 'link_right_knee_pitch', 
        'link_right_ankle_pitch', 'link_right_ankle_roll', 'head_link'
    ]
    
    # Create mapping (map source index to target index)
    mapping = {}
    reverse_mapping = {}  # For verification purposes
    for source_idx, source_name in enumerate(source_body_names):
        # Special case for the root joint
        if source_name == 'x1-body':
            target_name = 'base_link'
        else:
            target_name = source_name
        
        if target_name in target_body_names:
            target_idx = target_body_names.index(target_name)
            mapping[source_idx] = target_idx
            reverse_mapping[target_idx] = source_idx
    
    # Debug the mapping
    print("Mapping from source to target indices:")
    for src, tgt in sorted(mapping.items()):
        src_name = source_body_names[src]
        tgt_name = target_body_names[tgt]
        print(f"Source {src} ({src_name}) -> Target {tgt} ({tgt_name})")
    
    # Process each motion in the data
    remapped_data = {}
    for key, motion_data in all_data.items():
        print(f"Processing key: {key}")
        remapped_motion = motion_data.copy()
        
        # Remap DOF positions if present
        if 'dof' in motion_data:
            dof = motion_data['dof']
            print(f"Original dof shape: {dof.shape}")
            
            # Check if this is a batch of motions or a single motion
            if dof.ndim == 2:  # [time, dof]
                n_frames, n_dofs = dof.shape
                new_dof = np.zeros_like(dof)  # Same shape as original
                
                # Remap the DOFs (skip the root at index 0)
                for source_idx, target_idx in mapping.items():
                    if source_idx > 0 and target_idx > 0:  # Skip root joint
                        source_dof_idx = source_idx - 1
                        target_dof_idx = target_idx - 1
                        
                        if source_dof_idx < n_dofs and target_dof_idx < n_dofs:
                            new_dof[:, target_dof_idx] = dof[:, source_dof_idx]
                
                # Verify some example mappings
                sample_frame = min(5, n_frames-1)
                print("\nDOF Remapping Verification (sample frame):")
                # Check a few key joints to confirm remapping
                for joint_name, source_idx in [(name, idx) for idx, name in enumerate(source_body_names) if idx > 0]:
                    if source_idx in mapping and source_idx - 1 < n_dofs:
                        target_idx = mapping[source_idx]
                        if target_idx > 0 and target_idx - 1 < n_dofs:
                            print(f"Joint {joint_name}:")
                            print(f"  Source idx: {source_idx-1}, value: {dof[sample_frame, source_idx-1]}")
                            print(f"  Target idx: {target_idx-1}, value: {new_dof[sample_frame, target_idx-1]}")
                            # Ensure they match
                            assert np.allclose(dof[sample_frame, source_idx-1], new_dof[sample_frame, target_idx-1]), \
                                f"DOF values don't match for {joint_name}"
                
                remapped_motion['dof'] = new_dof
                print(f"Remapped dof shape: {new_dof.shape}")
            
            elif dof.ndim == 1:  # single frame [dof]
                new_dof = np.zeros_like(dof)  # Same shape as original
                
                # Remap the DOFs (skip the root at index 0)
                for source_idx, target_idx in mapping.items():
                    if source_idx > 0 and target_idx > 0:  # Skip root joint
                        source_dof_idx = source_idx - 1
                        target_dof_idx = target_idx - 1
                        
                        if source_dof_idx < len(dof) and target_dof_idx < len(dof):
                            new_dof[target_dof_idx] = dof[source_dof_idx]
                
                remapped_motion['dof'] = new_dof
                print(f"Remapped dof shape: {new_dof.shape}")
        
        # Remap pose_aa if present
        if 'pose_aa' in motion_data:
            pose_aa = motion_data['pose_aa']
            print(f"Original pose_aa shape: {pose_aa.shape}")
            
            # Handle different dimensions
            if pose_aa.ndim == 3:  # [time, joints, 3]
                n_frames, n_joints, axis_dims = pose_aa.shape
                new_pose_aa = np.zeros_like(pose_aa)  # Same shape as original
                
                print(f"Detected {n_joints} joints in pose_aa")
                
                # Directly map the 3D array without flattening
                for source_idx, target_idx in mapping.items():
                    if source_idx < n_joints and target_idx < n_joints:
                        new_pose_aa[:, target_idx, :] = pose_aa[:, source_idx, :]
                
                # Verify some example mappings
                sample_frame = min(5, n_frames-1)
                print("\nPose_AA Remapping Verification (sample frame):")
                
                # Check multiple joints to verify remapping
                sample_joints = [1, 7, 18, 24]  # Choose a few joints across different body parts
                for joint_idx in sample_joints:
                    if joint_idx in mapping and joint_idx < n_joints:
                        target_idx = mapping[joint_idx]
                        if target_idx < n_joints:
                            joint_name = source_body_names[joint_idx]
                            target_name = target_body_names[target_idx]
                            print(f"Joint {joint_idx} ({joint_name}) -> {target_idx} ({target_name}):")
                            print(f"  Original: {pose_aa[sample_frame, joint_idx, :]}")
                            print(f"  Remapped: {new_pose_aa[sample_frame, target_idx, :]}")
                            # Ensure they match
                            assert np.allclose(pose_aa[sample_frame, joint_idx, :], new_pose_aa[sample_frame, target_idx, :]), \
                                f"Pose values don't match for joint {joint_idx}"
                
                remapped_motion['pose_aa'] = new_pose_aa
                print(f"Remapped pose_aa shape: {new_pose_aa.shape}")
            
            elif pose_aa.ndim == 2:  # Flattened [time, joints*3]
                t, total_dims = pose_aa.shape
                joint_count = total_dims // 3
                
                print(f"Detected {joint_count} joints in flattened pose_aa")
                new_pose_aa = np.zeros_like(pose_aa)  # Same shape as original
                
                # Remap the pose data
                for source_idx, target_idx in mapping.items():
                    src_start = source_idx * 3
                    src_end = (source_idx + 1) * 3
                    tgt_start = target_idx * 3
                    tgt_end = (target_idx + 1) * 3
                    
                    if src_end <= total_dims and tgt_end <= total_dims:
                        new_pose_aa[:, tgt_start:tgt_end] = pose_aa[:, src_start:src_end]
                
                remapped_motion['pose_aa'] = new_pose_aa
                print(f"Remapped pose_aa shape: {new_pose_aa.shape}")
            
            else:
                print(f"Unsupported pose_aa dimensions: {pose_aa.ndim}")
        
        # Handle other joint-related fields if present
        if 'smpl_joints' in motion_data:
            print("Note: 'smpl_joints' field is present but not remapped - keeping original")
        
        remapped_data[key] = remapped_motion
    
    # Generate output path if not provided
    if output_pkl_path is None:
        dir_name = os.path.dirname(input_pkl_path)
        base_name = os.path.basename(input_pkl_path)
        output_pkl_path = os.path.join(dir_name, f"remapped_{base_name}")
    
    # Save the remapped data
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    joblib.dump(remapped_data, output_pkl_path)
    print(f"Saved remapped data to: {output_pkl_path}")
    
    return remapped_data

def main():
    parser = argparse.ArgumentParser(description='Remap PKL file body order to match simulator body list')
    parser.add_argument('--input', type=str, required=True, help='Path to input PKL file')
    parser.add_argument('--output', type=str, default=None, help='Path to output PKL file')
    
    args = parser.parse_args()
    
    remap_pkl_body_order(args.input, args.output)

if __name__ == "__main__":
    main()