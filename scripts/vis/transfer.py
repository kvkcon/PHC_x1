import mujoco
import xml.etree.ElementTree as ET

# Define file paths (adjust these to your specific URDF and MJCF paths)
urdf_path = "/home/bbw/ASAPx1/PHC_x1/phc/data/assets/robot/agibot_x1/x1_25dof.urdf"  # Replace with your URDF file path
mjcf_path = "/home/bbw/ASAPx1/PHC_x1/phc/data/assets/robot/agibot_x1/x1.xml"       # Replace with your desired MJCF output path

# Step 1: Parse the URDF file to extract joint types and effort limits
urdf_tree = ET.parse(urdf_path)
urdf_root = urdf_tree.getroot()

# Dictionaries to store joint information
joint_types = {}    # Stores joint type (e.g., "revolute", "fixed")
effort_limits = {}  # Stores effort limits (e.g., maximum torque or force)

for joint in urdf_root.findall(".//joint"):
    joint_name = joint.get("name")
    joint_type = joint.get("type")
    joint_types[joint_name] = joint_type
    limit = joint.find("limit")
    if limit is not None:
        effort = limit.get("effort")
        if effort is not None:
            effort_limits[joint_name] = float(effort)

# Step 2: Create and compile the MuJoCo model from the URDF
spec = mujoco.MjSpec()
spec.from_file(urdf_path)
model = spec.compile()

# Step 3: Convert the spec to XML and parse it
xml_str = spec.to_xml()
root = ET.fromstring(xml_str)

# Step 4: Add or find the actuator section
actuator_elem = root.find("actuator")
if actuator_elem is None:
    actuator_elem = ET.Element("actuator")
    root.append(actuator_elem)

# Step 5: Add motors with ctrllimited and ctrlrange for non-fixed joints
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if joint_name and joint_types.get(joint_name, "") != "fixed":
        motor_elem = ET.Element("motor", name=f"motor_{joint_name}", joint=joint_name)
        if joint_name in effort_limits:
            effort = effort_limits[joint_name]
            motor_elem.set("ctrllimited", "true")
            motor_elem.set("ctrlrange", f"{-effort} {effort}")
        # Note: If no effort limit exists, the motor is added without limits
        actuator_elem.append(motor_elem)

# Step 6: Write the modified XML to file
new_xml_str = ET.tostring(root, encoding="unicode")
with open(mjcf_path, 'w') as f:
    f.write(new_xml_str)

print(f"MJCF file saved successfully to {mjcf_path}!")