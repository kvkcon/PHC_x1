import xml.etree.ElementTree as ET
import os

# 添加ET.indent功能for Python 3.8 before
def indent(elem, level=0, space="  "):
    """为ElementTree添加缩进以美化XML输出"""
    i = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + space
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1, space)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def quat_to_rpy(quat):
    """
    将四元数转换为欧拉角（roll, pitch, yaw），这里简化为返回'0 0 0'。
    实际应用中需要使用数学库（如scipy）进行转换。
    """
    # 假设quat为'x y z w'顺序，MuJoCo默认'w x y z'，需要调整
    return '0 0 0'  # 占位，实际需要转换

def process_body(body, parent_link_name, urdf_root, mesh_dir='package://your_package/meshes/'):
    """
    递归处理MuJoCo的<body>元素，转换为URDF的<link>和<joint>。
    
    参数：
    - body: MuJoCo XML中的<body>元素
    - parent_link_name: 父link的名称
    - urdf_root: URDF的根元素
    - mesh_dir: mesh文件的路径前缀
    """
    # 获取当前body的名称作为link名称
    link_name = body.get('name')
    link = ET.SubElement(urdf_root, 'link', name=link_name)
    
    # 处理惯性<inertial>
    inertial = body.find('inertial')
    if inertial is not None:
        inertial_elem = ET.SubElement(link, 'inertial')
        pos = inertial.get('pos', '0 0 0')
        quat = inertial.get('quat', '1 0 0 0')  # 默认四元数
        mass = inertial.get('mass', '0')
        # fullinertia优先，否则使用diaginertia
        if 'fullinertia' in inertial.attrib:
            inertia_vals = inertial.get('fullinertia', '0 0 0 0 0 0').split()
            ixx, iyy, izz, ixy, ixz, iyz = inertia_vals
        else:
            diag = inertial.get('diaginertia', '0 0 0').split()
            ixx, iyy, izz = diag
            ixy = ixz = iyz = '0'
        
        rpy = quat_to_rpy(quat)
        ET.SubElement(inertial_elem, 'origin', xyz=pos, rpy=rpy)
        ET.SubElement(inertial_elem, 'mass', value=mass)
        ET.SubElement(inertial_elem, 'inertia', ixx=ixx, ixy=ixy, ixz=ixz, iyy=iyy, iyz=iyz, izz=izz)
    
    # 处理视觉<visual>和碰撞<collision>（假设geom同时用于两者）
    for geom in body.findall('geom'):
        if geom.get('type') == 'mesh':
            mesh_name = geom.get('mesh')
            mesh_filename = f"{mesh_dir}{mesh_name}.STL"
            rgba = geom.get('rgba', '1 1 1 1')
            pos = geom.get('pos', '0 0 0')
            euler = geom.get('euler', '0 0 0')
            
            # 添加visual
            visual = ET.SubElement(link, 'visual')
            ET.SubElement(visual, 'origin', xyz=pos, rpy=euler)
            geometry = ET.SubElement(visual, 'geometry')
            ET.SubElement(geometry, 'mesh', filename=mesh_filename)
            material = ET.SubElement(visual, 'material', name=f"{link_name}_color")
            ET.SubElement(material, 'color', rgba=rgba)
            
            # 添加collision（假设visual和collision相同）
            if geom.get('class') == 'collision' or True:  # 简化为总是添加collision
                collision = ET.SubElement(link, 'collision')
                ET.SubElement(collision, 'origin', xyz=pos, rpy=euler)
                collision_geometry = ET.SubElement(collision, 'geometry')
                ET.SubElement(collision_geometry, 'mesh', filename=mesh_filename)
    
    # 处理关节<joint>
    joint_elem = body.find('joint')
    if joint_elem is not None and joint_elem.get('type') != 'free':  # 跳过freejoint
        joint_name = joint_elem.get('name')
        joint_type = 'revolute' if joint_elem.get('type') == 'hinge' else joint_elem.get('type')
        joint = ET.SubElement(urdf_root, 'joint', name=joint_name, type=joint_type)
        
        ET.SubElement(joint, 'parent', link=parent_link_name)
        ET.SubElement(joint, 'child', link=link_name)
        
        # joint的origin使用body的pos和quat
        body_pos = body.get('pos', '0 0 0')
        body_quat = body.get('quat', '1 0 0 0')
        rpy = quat_to_rpy(body_quat)
        ET.SubElement(joint, 'origin', xyz=body_pos, rpy=rpy)
        
        axis = joint_elem.get('axis', '0 0 1')
        ET.SubElement(joint, 'axis', xyz=axis)
        
        range = joint_elem.get('range')
        if range:
            lower, upper = range.split()
            ET.SubElement(joint, 'limit', lower=lower, upper=upper, effort='150', velocity='10')
        damping = joint_elem.get('damping', '0')
        ET.SubElement(joint, 'dynamics', damping=damping)
    
    # 递归处理子body
    for child_body in body.findall('body'):
        process_body(child_body, link_name, urdf_root, mesh_dir)

def convert_mujoco_to_urdf(mujoco_path, urdf_path, mesh_dir):
    """
    主函数，将MuJoCo XML转换为URDF。
    
    参数：
    - mujoco_path: MuJoCo XML文件路径
    - urdf_path: 输出URDF文件路径
    """
    # 解析MuJoCo XML
    tree = ET.parse(mujoco_path)
    root = tree.getroot()
    
    # 创建URDF根元素
    urdf_root = ET.Element('robot', name='zhiyuan')
    
    # 找到机器人模型的根body
    robot_body = root.find('.//body[@name="x1-body"]')
    if robot_body is None:
        raise ValueError("未找到name='x1-body'的根body")
    
    # 处理浮动基座：创建一个base_link，不连接world
    process_body(robot_body, None, urdf_root, mesh_dir)
    
    # 美化XML输出（可选）
    indent(urdf_root, space="  ")
    # ET.indent(urdf_root, space="  ")
    
    # 写入URDF文件
    urdf_tree = ET.ElementTree(urdf_root)
    urdf_tree.write(urdf_path, encoding='utf-8', xml_declaration=True)
    print(f"URDF文件已生成：{urdf_path}")

if __name__ == "__main__":
    # 请替换为您的文件路径
    mujoco_xml_path = 'C:\\Users\\coconerd\\Documents\\GitHub\\ASAPx1\\PHC_x1\\phc\\data\\assets\\robot\\zhiyuan_x1\\x1.xml'
    urdf_output_path = 'C:\\Users\\coconerd\\Documents\\GitHub\\ASAPx1\\PHC_x1\\phc\\data\\assets\\robot\\zhiyuan_x1\\x1_grok.urdf'
    mesh_dir = './meshes/' # package://../meshes/
    
    # 执行转换
    convert_mujoco_to_urdf(mujoco_xml_path, urdf_output_path, mesh_dir)