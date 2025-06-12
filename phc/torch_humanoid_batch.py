import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
import torch 
from collections import defaultdict
import torch.nn.functional as F
from typing import Optional, Union

import numpy as np
# import smpl_sim.utils.rotation_conversions as tRot
# import phc.utils.rotation_conversions as tRot
from scipy.spatial.transform import Rotation as sRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import smpl_sim.poselib.core.rotation3d as pRot
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
import copy
from collections import OrderedDict
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from stl import mesh
import logging
import open3d as o3d

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Rotation conversion functions (extracted from rotation_conversions.py)
def wxyz_to_xyzw(quat):
    """Convert quaternion from w,x,y,z to x,y,z,w format"""
    return quat[..., [1, 2, 3, 0]]

def xyzw_to_wxyz(quat):
    """Convert quaternion from x,y,z,w to w,x,y,z format"""
    return quat[..., [3, 0, 1, 2]]

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
                          ].reshape(batch_dim + (4,))

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (torch.sin(half_angles[~small_angles]) / angles[~small_angles])
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions

class Humanoid_Batch:

    def __init__(self, cfg, device = torch.device("cpu")):
        self.cfg = cfg
        self.mjcf_file = cfg.asset.assetFileName
        
        # Safely check if extend_config exists in the YAML configuration
        if not OmegaConf.select(cfg, 'extend_config'):
            # If not in YAML, create an empty list using OmegaConf.create
            OmegaConf.set_struct(cfg, False)  # Temporarily disable struct mode
            cfg.extend_config = []
            OmegaConf.set_struct(cfg, True)   # Re-enable struct mode
        
        parser = XMLParser(remove_blank_text=True)
        tree = parse(BytesIO(open(self.mjcf_file, "rb").read()), parser=parser,)
        
        # Get all actuated joints from motors
        motors = sorted([m.attrib['joint'] for m in tree.getroot().find("actuator").findall('.//motor')])
        assert(len(motors) > 0, "No motors found in the mjcf file")
        
        all_joints = []
        fixed_joints = []
        for j in tree.getroot().find("worldbody").findall('.//joint'):
            joint_name = j.attrib['name']
            all_joints.append(joint_name)
            if j.attrib.get('type') == 'fixed':
                fixed_joints.append(joint_name)
        
        self.num_dof = len(motors)  # Should be 23 as per the XML
        self.num_extend_dof = self.num_dof
        self.fixed_joints = fixed_joints  # record fixed joints
        
        self.mjcf_data = mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data['node_names'])
        print(f"self.body_names={self.body_names}")
        self._parents = mjcf_data['parent_indices']
        self.body_names_augment = copy.deepcopy(mjcf_data['node_names'])
        self._offsets = mjcf_data['local_translation'][None, ].to(device)
        self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
        
        print("-------------------------body_list----------from PHC_x1\\phc\\utils\\torch_humanoid_batch.py--------------------------------")
        print(f"self.body_names={self.body_names}")
        self.body_index_list = np.array([self.body_names.index(name) for name in self.body_names])
        print(f"self.body_index_list={self.body_index_list}")

                
        # get actuated_joints_idx
        print(f"mjcf_data['body_to_joint']={mjcf_data['body_to_joint']}")
        fixed_joints_idx = {}
        for i, name in enumerate(self.body_names):
            if i==0: continue
            if name not in mjcf_data['body_to_joint']:
                print(f"Joint {name} not found in body_to_joint")
                fixed_joints_idx[name] = i
        self.fixed_joints_idx = np.array([self.body_names.index(k) for k,v in fixed_joints_idx.items()])
        print(f"self.fixed_joints_idx={self.fixed_joints_idx}")
        self.actuated_joints_idx = np.array([self.body_names.index(k) for k, v in mjcf_data['body_to_joint'].items() 
                                            if v in motors])
        
        # # get fixed_joints_idx
        # self.fixed_joints_idx = np.array([self.body_names.index(k) for k, v in mjcf_data['body_to_joint'].items() 
        #                                 if v in fixed_joints])
        
        # create joint_name_to_idx
        self.joint_name_to_idx = {name: i for i, name in enumerate(self.body_names)}
        
        # Create a mapping of joint names to their axis information
        joint_to_axis = {}
        for j in tree.getroot().find("worldbody").findall('.//joint'):
            if 'axis' in j.attrib:
                joint_to_axis[j.attrib['name']] = [float(i) for i in j.attrib['axis'].split()]
        
        # Check if there's a freejoint (floating base)
        freejoint = tree.getroot().find("worldbody").find('.//freejoint')
        self.has_freejoint = freejoint is not None
        
        # Build dof_axis array based on motors (actuated joints)
        self.dof_axis = []
        self.dof_axis_names = []
        for dof_axis_k, dof_axis_v in joint_to_axis.items():
            if dof_axis_k in motors:
                self.dof_axis.append(dof_axis_v)
                self.dof_axis_names.append(dof_axis_k)
        
        self.dof_axis = torch.tensor(self.dof_axis)
        print(f"self.dof_axis_names={self.dof_axis_names}")


        if len(cfg.extend_config) > 0: 
            for extend_config in cfg.extend_config:
                self.body_names_augment += [extend_config.joint_name]
                self._parents = torch.cat([self._parents, torch.tensor([self.body_names.index(extend_config.parent_name)]).to(device)], dim = 0)
                self._offsets = torch.cat([self._offsets, torch.tensor([[extend_config.pos]]).to(device)], dim = 1)
                self._local_rotation = torch.cat([self._local_rotation, torch.tensor([[extend_config.rot]]).to(device)], dim = 1)
                self.num_extend_dof += 1
            
        self.num_bodies = len(self.body_names)
        self.num_bodies_augment = len(self.body_names_augment)
        
        # print   debug code
        print(f"Total bodies: {self.num_bodies}, Actuated joints: {len(self.actuated_joints_idx)}, Fixed joints: {len(self.fixed_joints_idx)}")
        print(f"DOF names from config: {cfg.dof_names}")
        print(f"Actuated joints: {[self.body_names[i] for i in self.actuated_joints_idx]}")
        print(f"Fixed joints: {[self.body_names[i] for i in self.fixed_joints_idx]}")

        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z
        self.load_mesh()
        
    def from_mjcf(self, path):
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint") # joints need to remove the first 6 joints
            if len(all_joints) == 6:
                all_joints = all_joints[6:]
            
            # 添加用于追踪的列表
            joint_names_for_range = []
            joint_indices_for_range = []
            
            for joint in all_joints:
                #check fixed joint
                joint_type = joint.attrib.get("type", "hinge")
                if joint_type == "fixed":
                    continue
                    
                joint_name = joint.attrib.get("name")
                joint_names_for_range.append(joint_name)
                joint_indices_for_range.append(curr_index)
                
                if not joint.attrib.get("range") is None: 
                    joint_range = np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                    joints_range.append(joint_range)
                    print(f"Joint range added: {joint_name} (body: {node_name}, body_index: {curr_index}) -> range: {joint_range}")
                else:
                    if not joint_type == "free":
                        default_range = [-np.pi, np.pi]
                        joints_range.append(default_range)
                        print(f"Joint range added: {joint_name} (body: {node_name}, body_index: {curr_index}) -> range: {default_range} (default)")
                        
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name")
                
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        print(f"len(joint_range)={len(joints_range)}, num_dof={self.num_dof}")
        assert(len(joints_range) == self.num_dof) 
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint
        }

        
    def fk_batch(self, pose, trans, convert_to_mat=True, return_full = False, dt=1/30):
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[..., :len(self._parents), :] # H1 fitted joints might have extra joints
        
        if convert_to_mat:
            pose_quat = axis_angle_to_quaternion(pose.clone())
            pose_mat = quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
            
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        
        wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))
        if len(self.cfg.extend_config) > 0:
            if return_full:
                return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt) 
                return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)
                
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot
            
            wbody_pos = wbody_pos[..., :self.num_bodies, :]
            wbody_mat = wbody_mat[..., :self.num_bodies, :, :]
            wbody_rot = wbody_rot[..., :self.num_bodies, :]

        
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)  # Isaac gym is [x, y, z, w]. All the previous functions are [w, x, y, z]
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            return_dict.local_rotation = wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            if len(self.cfg.extend_config) > 0:
                return_dict.dof_pos = pose.sum(dim = -1)[..., 1:self.num_bodies] # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter. 
            else:
                if not len(self.actuated_joints_idx) == len(self.body_names):
                    return_dict.dof_pos = pose.sum(dim = -1)[..., self.actuated_joints_idx]
                else:
                    return_dict.dof_pos = pose.sum(dim = -1)[..., 1:]
            
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1] )/dt)
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim = 1)
            return_dict.fps = int(1/dt)
        
        return return_dict
    
    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of actuated joints):
        -- rotations: (B, seq_len, J+1, 3, 3) tensor of rotation matrices for non-root joints
        -- root_rotations: (B, seq_len, 1, 3, 3) tensor of rotation matrices for the root joint
        -- root_positions: (B, seq_len, 3) tensor describing the root joint positions
        Output: joint positions (B, seq_len, J, 3) and rotations (B, seq_len, J, 3, 3)
        """
        
        # print(f"rotations.shape={rotations.shape}, root_rotations.shape={root_rotations.shape}, root_positions.shape={root_positions.shape}")
        # print(f"rotations={rotations}, root_rotations={root_rotations}, root_positions={root_positions}")

        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        # print(f"J={J}, self._offsets.shape={self._offsets.shape}, self._parents.shape={self._parents.shape}")
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))
        
        # create fixed joint rot3*3
        identity_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        identity_rot = identity_rot.expand(B, seq_len, 1, 3, 3)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                # print(f"i={i}, parent={self._parents[i]}, rotations_world[parent].shape={rotations_world[self._parents[i]].shape}")
                parent_rot = rotations_world[self._parents[i]]
                
                # count pos
                jpos = (torch.matmul(parent_rot[:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + 
                        positions_world[self._parents[i]])
                positions_world.append(jpos)
                
                # handle rot - check whether fixed
                if i in self.fixed_joints_idx:
                    # for fixed joint，only using parents' rot
                    rotations_world.append(parent_rot)
                else:
                    # check whether index valid
                    if i - 1 < rotations.shape[2]:
                        rot_mat = torch.matmul(parent_rot, 
                                torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], 
                                        rotations[:, :, (i - 1):i, :]))
                        rotations_world.append(rot_mat)
                    else:
                        # using identity_rot if index in invalid
                        rotations_world.append(identity_rot)
        
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        if guassian_filter:
            velocity = torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)
        else:
            velocity = torch.from_numpy(velocity).to(p)
        
        return velocity
    
    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis
        diff_quat_data = pRot.quat_identity_like(r).to(r)
        diff_quat_data[..., :-1, :, :] = pRot.quat_mul_norm(r[..., 1:, :, :], pRot.quat_inverse(r[..., :-1, :, :]))
        diff_angle, diff_axis = pRot.quat_angle_axis(diff_quat_data)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        if guassian_filter:
            angular_velocity = torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"),)
        return angular_velocity  
    
    def load_mesh(self):
        xml_base = os.path.dirname(self.mjcf_file)
        # Read the compiler tag from the g1.xml file to find if there is a meshdir defined
        tree = ETree.parse(self.mjcf_file)
        xml_doc_root = tree.getroot()
        compiler_tag = xml_doc_root.find("compiler")
        
        if compiler_tag is not None and "meshdir" in compiler_tag.attrib:
            mesh_base = os.path.join(xml_base, compiler_tag.attrib["meshdir"])
        else:
            mesh_base = xml_base
            
        self.tree = tree = ETree.parse(self.mjcf_file)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")

        xml_assets = xml_doc_root.find("asset")
        all_mesh = xml_assets.findall(".//mesh")

        geoms = xml_world_body.findall(".//geom")

        all_joints = xml_world_body.findall(".//joint")
        all_motors = tree.findall(".//motor")
        all_bodies = xml_world_body.findall(".//body")

        def find_parent(root, child):
            for parent in root.iter():
                for elem in parent:
                    if elem == child:
                        return parent
            return None

        mesh_dict = {}
        mesh_parent_dict = {}
        
        
        for mesh_file_node in tqdm(all_mesh):
            mesh_name = mesh_file_node.attrib["name"]
            mesh_file = mesh_file_node.attrib["file"]
            mesh_full_file = osp.join(mesh_base, mesh_file)
            mesh_obj = o3d.io.read_triangle_mesh(mesh_full_file)
            mesh_dict[mesh_name] = mesh_obj

        geom_transform = {}
        
        body_to_mesh = defaultdict(set)
        mesh_to_body = {}
        for geom_node in tqdm(geoms):
            if 'mesh' in geom_node.attrib: 
                parent = find_parent(xml_doc_root, geom_node)
                body_to_mesh[parent.attrib['name']].add(geom_node.attrib['mesh'])
                mesh_to_body[geom_node] = parent
                if "pos" in geom_node.attrib or "quat" in geom_node.attrib:
                    geom_transform[parent.attrib['name']] = {}
                    geom_transform[parent.attrib['name']]["pos"] = np.array([0.0, 0.0, 0.0])
                    geom_transform[parent.attrib['name']]["quat"] = np.array([1.0, 0.0, 0.0, 0.0])
                    if "pos" in geom_node.attrib:
                        geom_transform[parent.attrib['name']]["pos"] = np.array([float(f) for f in geom_node.attrib['pos'].split(" ")])
                    if "quat" in geom_node.attrib:
                        geom_transform[parent.attrib['name']]["quat"] = np.array([float(f) for f in geom_node.attrib['quat'].split(" ")])
                    
            else:
                pass
            
        self.geom_transform = geom_transform
        self.mesh_dict = mesh_dict
        self.body_to_mesh = body_to_mesh
        self.mesh_to_body = mesh_to_body

    def mesh_fk(self, pose = None, trans = None):
        """
        Load the mesh from the XML file and merge them into the humanoid based on the current pose.
        """
        if pose is None:
            fk_res = self.fk_batch(torch.zeros(1, 1, len(self.body_names_augment), 3), torch.zeros(1, 1, 3))
        else:
            fk_res = self.fk_batch(pose, trans)
        
        g_trans = fk_res.global_translation.squeeze()
        g_rot = fk_res.global_rotation_mat.squeeze()
        geoms = self.tree.find("worldbody").findall(".//geom")
        joined_mesh_obj = []
        for geom in geoms:
            if 'mesh' not in geom.attrib:
                continue
            parent_name = geom.attrib['mesh']
            

            k = self.mesh_to_body[geom].attrib['name']
            mesh_names = self.body_to_mesh[k]
            body_idx = self.body_names.index(k)
            
            body_trans = g_trans[body_idx].numpy().copy()
            body_rot = g_rot[body_idx].numpy().copy()
            for mesh_name in mesh_names:
                mesh_obj = copy.deepcopy(self.mesh_dict[mesh_name])
                if k in self.geom_transform:
                    pos = self.geom_transform[k]['pos']
                    quat = self.geom_transform[k]['quat']
                    body_trans = body_trans + body_rot @ pos
                    global_rot =  (body_rot   @ sRot.from_quat(quat[[1, 2, 3, 0]]).as_matrix()).T
                else:
                    global_rot = body_rot.T
                mesh_obj.rotate(global_rot.T, center=(0, 0, 0))
                mesh_obj.translate(body_trans)
                joined_mesh_obj.append(mesh_obj)
                
        # Merge all meshes into a single mesh
        merged_mesh = joined_mesh_obj[0]
        for mesh in joined_mesh_obj[1:]:
            merged_mesh += mesh
        
        # Save the merged mesh to a file
        # merged_mesh.compute_vertex_normals()
        # o3d.io.write_triangle_mesh(f"data/{self.cfg.humanoid_type}/combined_{self.cfg.humanoid_type}.stl", merged_mesh)
        return merged_mesh

    
    
@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cpu")
    humanoid_fk = Humanoid_Batch(cfg.robot, device)
    humanoid_fk.mesh_fk()

if __name__ == "__main__":
    main()