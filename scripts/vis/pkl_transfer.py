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
    
    # Define the mapping from source order to target order
    # Source order (from PHC_x1\phc\utils\torch_humanoid_batch.py)
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
    
    # Target order (from simulator._body_list)
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
    # Note: 'x1-body' in source maps to 'base_link' in target, and we need to handle 'head_link'
    mapping = {}
    for source_idx, source_name in enumerate(source_body_names):
        if source_name == 'x1-body':
            target_name = 'base_link'
        else:
            target_name = source_name
        
        if target_name in target_body_names:
            target_idx = target_body_names.index(target_name)
            mapping[source_idx] = target_idx
    
    # Process each motion in the data
    remapped_data = {}
    for key, motion_data in all_data.items():
        remapped_motion = motion_data.copy()
        
        # Remap DOF positions if present
        if 'dof' in motion_data:
            dof = motion_data['dof']
            # Check if this is a batch of motions or a single motion
            if dof.ndim == 2:  # [time, dof]
                n_frames, n_dofs = dof.shape
                new_dof = np.zeros((n_frames, len(target_body_names) - 1))  # -1 because the root doesn't have DOFs
                
                for source_idx, target_idx in mapping.items():
                    if source_idx > 0 and target_idx > 0:  # Skip root joint
                        new_dof[:, target_idx-1] = dof[:, source_idx-1]
                
                remapped_motion['dof'] = new_dof
            elif dof.ndim == 1:  # single frame [dof]
                new_dof = np.zeros(len(target_body_names) - 1)
                
                for source_idx, target_idx in mapping.items():
                    if source_idx > 0 and target_idx > 0:  # Skip root joint
                        new_dof[target_idx-1] = dof[source_idx-1]
                
                remapped_motion['dof'] = new_dof
        
        # Remap pose_aa if present (this contains axis-angle representations)
        if 'pose_aa' in motion_data:
            pose_aa = motion_data['pose_aa']
            # Handle different dimensions
            if pose_aa.ndim == 3:  # [batch, time, joints*3]
                b, t, _ = pose_aa.shape
                new_pose_aa = np.zeros((b, t, len(target_body_names) * 3))
                
                for source_idx, target_idx in mapping.items():
                    new_pose_aa[:, :, target_idx*3:(target_idx+1)*3] = pose_aa[:, :, source_idx*3:(source_idx+1)*3]
                
                # Add zero vector for the head joint if it exists in target but not source
                if len(target_body_names) > len(source_body_names):
                    head_idx = target_body_names.index('head_link')
                    new_pose_aa[:, :, head_idx*3:(head_idx+1)*3] = 0
                
                remapped_motion['pose_aa'] = new_pose_aa
            
            elif pose_aa.ndim == 2:  # [time, joints*3]
                t, _ = pose_aa.shape
                new_pose_aa = np.zeros((t, len(target_body_names) * 3))
                
                for source_idx, target_idx in mapping.items():
                    new_pose_aa[:, target_idx*3:(target_idx+1)*3] = pose_aa[:, source_idx*3:(source_idx+1)*3]
                
                # Add zero vector for the head joint
                if len(target_body_names) > len(source_body_names):
                    head_idx = target_body_names.index('head_link')
                    new_pose_aa[:, head_idx*3:(head_idx+1)*3] = 0
                
                remapped_motion['pose_aa'] = new_pose_aa
        
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