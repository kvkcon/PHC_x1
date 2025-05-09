# import xml.etree.ElementTree as ET

# # Path to the XML file
# xml_path = '/home/bbw/ASAPx1/PHC_x1/phc/data/assets/robot/agibot_x1/x1.xml'

# # Parse the XML file
# tree = ET.parse(xml_path)
# root = tree.getroot()

# # Extract body names
# def get_body_names(root):
#     body_names = []
#     for body in root.findall('.//body'):
#         name = body.get('name')
#         if name:
#             body_names.append(name)
#     return body_names

# # Extract DOF names
# def get_dof_names(root):
#     dof_names = []
#     for joint in root.findall('.//joint'):
#         name = joint.get('name')
#         if name:
#             dof_names.append(name)
#     return dof_names

# # Print results
# print("Body Names:")
# body_names = get_body_names(root)
# for name in body_names:
#     print(name)

# print("\nDOF Names:")
# dof_names = get_dof_names(root)
# for name in dof_names:
#     print(name)

# # For limb_weight_group, you'll need to manually categorize based on the body names
# print("\nSuggested Limb Weight Group Categories:")
# left_lower_limb = [name for name in body_names if 'left' in name.lower() and ('hip' in name.lower() or 'knee' in name.lower() or 'ankle' in name.lower())]
# right_lower_limb = [name for name in body_names if 'right' in name.lower() and ('hip' in name.lower() or 'knee' in name.lower() or 'ankle' in name.lower())]
# torso = [name for name in body_names if 'base' in name.lower() or 'body' in name.lower()]
# left_upper_limb = [name for name in body_names if 'left' in name.lower() and ('shoulder' in name.lower() or 'elbow' in name.lower())]
# right_upper_limb = [name for name in body_names if 'right' in name.lower() and ('shoulder' in name.lower() or 'elbow' in name.lower())]

# print("Left Lower Limb:", left_lower_limb)
# print("Right Lower Limb:", right_lower_limb)
# print("Torso:", torso)
# print("Left Upper Limb:", left_upper_limb)
# print("Right Upper Limb:", right_upper_limb)

import mujoco
import argparse

def extract_model_info(mjcf_path):
    model = mujoco.MjModel.from_xml_path(mjcf_path)

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) 
                  for i in range(model.nbody) 
                  if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)]

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                   for i in range(model.njnt) 
                   if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)]

    print("Body Names:", body_names)
    print("Total Body Names:", len(body_names))
    print("DOF Names (Joint Names):", joint_names)
    print("Total DOF Names:", len(joint_names))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract body and joint names from a MuJoCo MJCF file.")
    parser.add_argument('--mjcf_path', type=str, default='/home/bbw/ASAPx1/PHC_x1/phc/data/assets/robot/zhiyuan_x1/x1_25dof_march.xml',
                        help="Path to the MJCF file. Default is '/home/bbw/ASAPx1/PHC_x1/phc/data/assets/robot/zhiyuan_x1/x1_25dof_march.xml'.")
    args = parser.parse_args()
    extract_model_info(args.mjcf_path)
