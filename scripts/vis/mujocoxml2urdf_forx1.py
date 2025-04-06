#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
import numpy as np
import argparse

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

def quaternion_to_euler(quat):
    """将四元数转换为欧拉角(RPY)"""
    # 假设四元数格式为(w, x, y, z)
    w, x, y, z = quat
    
    # Roll (绕X轴)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (绕Y轴)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # 使用90度，如果超出范围
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (绕Z轴)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return (roll, pitch, yaw)

def parse_pos(pos_str):
    """解析位置字符串为XYZ值"""
    if pos_str:
        return [float(x) for x in pos_str.split()]
    return [0, 0, 0]

def parse_quat(quat_str):
    """解析四元数字符串"""
    if quat_str:
        return [float(x) for x in quat_str.split()]
    return [1, 0, 0, 0]  # 默认四元数

def parse_euler(euler_str):
    """解析欧拉角字符串"""
    if euler_str:
        return [float(x) for x in euler_str.split()]
    return [0, 0, 0]  # 默认欧拉角

def joint_type_conversion(mujoco_type):
    """将MuJoCo关节类型转换为URDF关节类型"""
    conversion = {
        "hinge": "revolute",
        "slide": "prismatic",
        "ball": "spherical",  # URDF不直接支持球关节
        "free": "floating"    # URDF不直接支持自由关节
    }
    return conversion.get(mujoco_type, "fixed")

def convert_mujoco_to_urdf(mujoco_path, urdf_path, mesh_dir="meshes"):
    """将MuJoCo XML文件转换为URDF格式"""
    # 解析MuJoCo XML
    mujoco_tree = ET.parse(mujoco_path)
    mujoco_root = mujoco_tree.getroot()
    
    # 创建URDF根元素
    urdf_root = ET.Element("robot")
    model_name = mujoco_root.get("model", "converted_model")
    urdf_root.set("name", model_name)
    
    # 处理materials
    materials = {}
    # 可以在这里添加材质处理逻辑
    
    # 处理meshes
    meshes = {}
    for mesh in mujoco_root.findall(".//asset/mesh"):
        mesh_name = mesh.get("name")
        mesh_file = mesh.get("file")
        meshes[mesh_name] = mesh_file
    
    # 创建链接和关节字典
    links = {}
    joints = {}
    
    # 处理world body和其他bodies
    world_body = mujoco_root.find(".//worldbody")
    
    # 创建一个虚拟的根链接
    base_link = ET.SubElement(urdf_root, "link")
    base_link.set("name", "base_link")
    links["base_link"] = base_link
    
    # 递归处理body树
    def process_body(mujoco_body, parent_link_name="base_link"):
        body_name = mujoco_body.get("name", f"link_{len(links)}")
        
        # 创建一个新的URDF链接
        link = ET.SubElement(urdf_root, "link")
        link.set("name", body_name)
        links[body_name] = link
        
        # 处理惯性属性
        inertial = mujoco_body.find("inertial")
        if inertial is not None:
            urdf_inertial = ET.SubElement(link, "inertial")
            
            # 质量
            mass = inertial.get("mass")
            if mass:
                ET.SubElement(urdf_inertial, "mass").set("value", mass)
            
            # 惯性原点
            inertial_pos = parse_pos(inertial.get("pos", "0 0 0"))
            inertial_origin = ET.SubElement(urdf_inertial, "origin")
            inertial_origin.set("xyz", f"{inertial_pos[0]} {inertial_pos[1]} {inertial_pos[2]}")
            
            # 处理四元数或欧拉角
            if inertial.get("quat"):
                quat = parse_quat(inertial.get("quat"))
                rpy = quaternion_to_euler(quat)
                inertial_origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
            elif inertial.get("euler"):
                euler = parse_euler(inertial.get("euler"))
                inertial_origin.set("rpy", f"{euler[0]} {euler[1]} {euler[2]}")
            else:
                inertial_origin.set("rpy", "0 0 0")
            
            # 惯性张量
            if inertial.get("fullinertia"):
                ixx, iyy, izz, ixy, ixz, iyz = [float(x) for x in inertial.get("fullinertia").split()]
                inertia = ET.SubElement(urdf_inertial, "inertia")
                inertia.set("ixx", str(ixx))
                inertia.set("iyy", str(iyy))
                inertia.set("izz", str(izz))
                inertia.set("ixy", str(ixy))
                inertia.set("ixz", str(ixz))
                inertia.set("iyz", str(iyz))
            elif inertial.get("diaginertia"):
                ixx, iyy, izz = [float(x) for x in inertial.get("diaginertia").split()]
                inertia = ET.SubElement(urdf_inertial, "inertia")
                inertia.set("ixx", str(ixx))
                inertia.set("iyy", str(iyy))
                inertia.set("izz", str(izz))
                inertia.set("ixy", "0")
                inertia.set("ixz", "0")
                inertia.set("iyz", "0")
        
        # 处理视觉和碰撞几何体
        for geom in mujoco_body.findall("geom"):
            geom_type = geom.get("type", "mesh")
            geom_mesh = geom.get("mesh")
            geom_size = geom.get("size")
            geom_rgba = geom.get("rgba", "0.7 0.7 0.7 1")
            geom_pos = parse_pos(geom.get("pos", "0 0 0"))
            
            # 创建视觉元素
            visual = ET.SubElement(link, "visual")
            visual_origin = ET.SubElement(visual, "origin")
            visual_origin.set("xyz", f"{geom_pos[0]} {geom_pos[1]} {geom_pos[2]}")
            
            if geom.get("quat"):
                quat = parse_quat(geom.get("quat"))
                rpy = quaternion_to_euler(quat)
                visual_origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
            elif geom.get("euler"):
                euler = parse_euler(geom.get("euler"))
                visual_origin.set("rpy", f"{euler[0]} {euler[1]} {euler[2]}")
            else:
                visual_origin.set("rpy", "0 0 0")
            
            # 处理几何体类型
            visual_geometry = ET.SubElement(visual, "geometry")
            if geom_type == "mesh" and geom_mesh in meshes:
                mesh_elem = ET.SubElement(visual_geometry, "mesh")
                mesh_elem.set("filename", f"package://{mesh_dir}/{meshes[geom_mesh]}")
                # 如果有scale处理
            elif geom_type == "box" and geom_size:
                size_values = [float(x) for x in geom_size.split()]
                box_elem = ET.SubElement(visual_geometry, "box")
                box_elem.set("size", f"{2*size_values[0]} {2*size_values[1]} {2*size_values[2]}")
            elif geom_type == "sphere" and geom_size:
                sphere_elem = ET.SubElement(visual_geometry, "sphere")
                sphere_elem.set("radius", geom_size.split()[0])
            elif geom_type == "cylinder" and geom_size:
                size_values = [float(x) for x in geom_size.split()]
                cylinder_elem = ET.SubElement(visual_geometry, "cylinder")
                cylinder_elem.set("radius", str(size_values[0]))
                cylinder_elem.set("length", str(2*size_values[1]))
            elif geom_type == "capsule" and geom_size:
                # URDF没有胶囊体，用圆柱体近似
                size_values = [float(x) for x in geom_size.split()]
                cylinder_elem = ET.SubElement(visual_geometry, "cylinder")
                cylinder_elem.set("radius", str(size_values[0]))
                cylinder_elem.set("length", str(2*size_values[1]))
            
            # 处理材质
            material = ET.SubElement(visual, "material")
            material.set("name", f"material_{len(materials)}")
            color = ET.SubElement(material, "color")
            color.set("rgba", geom_rgba)
            materials[f"material_{len(materials)}"] = geom_rgba
            
            # 创建碰撞元素 (简单复制视觉元素的几何体)
            collision = ET.SubElement(link, "collision")
            collision_origin = ET.SubElement(collision, "origin")
            collision_origin.set("xyz", f"{geom_pos[0]} {geom_pos[1]} {geom_pos[2]}")
            
            if geom.get("quat"):
                quat = parse_quat(geom.get("quat"))
                rpy = quaternion_to_euler(quat)
                collision_origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
            elif geom.get("euler"):
                euler = parse_euler(geom.get("euler"))
                collision_origin.set("rpy", f"{euler[0]} {euler[1]} {euler[2]}")
            else:
                collision_origin.set("rpy", "0 0 0")
            
            # 复制几何体
            collision_geometry = ET.SubElement(collision, "geometry")
            if geom_type == "mesh" and geom_mesh in meshes:
                mesh_elem = ET.SubElement(collision_geometry, "mesh")
                mesh_elem.set("filename", f"package://{mesh_dir}/{meshes[geom_mesh]}")
            elif geom_type == "box" and geom_size:
                size_values = [float(x) for x in geom_size.split()]
                box_elem = ET.SubElement(collision_geometry, "box")
                box_elem.set("size", f"{2*size_values[0]} {2*size_values[1]} {2*size_values[2]}")
            elif geom_type == "sphere" and geom_size:
                sphere_elem = ET.SubElement(collision_geometry, "sphere")
                sphere_elem.set("radius", geom_size.split()[0])
            elif geom_type == "cylinder" and geom_size:
                size_values = [float(x) for x in geom_size.split()]
                cylinder_elem = ET.SubElement(collision_geometry, "cylinder")
                cylinder_elem.set("radius", str(size_values[0]))
                cylinder_elem.set("length", str(2*size_values[1]))
            elif geom_type == "capsule" and geom_size:
                size_values = [float(x) for x in geom_size.split()]
                cylinder_elem = ET.SubElement(collision_geometry, "cylinder")
                cylinder_elem.set("radius", str(size_values[0]))
                cylinder_elem.set("length", str(2*size_values[1]))
        
        # 处理关节
        joint = mujoco_body.find("joint")
        if joint is not None:
            joint_name = joint.get("name", f"joint_{len(joints)}")
            joint_type = joint_type_conversion(joint.get("type", "fixed"))
            
            urdf_joint = ET.SubElement(urdf_root, "joint")
            urdf_joint.set("name", joint_name)
            urdf_joint.set("type", joint_type)
            
            # 设置父子链接
            parent = ET.SubElement(urdf_joint, "parent")
            parent.set("link", parent_link_name)
            
            child = ET.SubElement(urdf_joint, "child")
            child.set("link", body_name)
            
            # 处理关节原点
            body_pos = parse_pos(mujoco_body.get("pos", "0 0 0"))
            joint_origin = ET.SubElement(urdf_joint, "origin")
            joint_origin.set("xyz", f"{body_pos[0]} {body_pos[1]} {body_pos[2]}")
            
            if mujoco_body.get("quat"):
                quat = parse_quat(mujoco_body.get("quat"))
                rpy = quaternion_to_euler(quat)
                joint_origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
            elif mujoco_body.get("euler"):
                euler = parse_euler(mujoco_body.get("euler"))
                joint_origin.set("rpy", f"{euler[0]} {euler[1]} {euler[2]}")
            else:
                joint_origin.set("rpy", "0 0 0")
            
            # 处理关节轴
            if joint_type in ["revolute", "prismatic", "continuous"]:
                axis = joint.get("axis", "0 0 1")
                urdf_axis = ET.SubElement(urdf_joint, "axis")
                urdf_axis.set("xyz", axis)
            
            # 处理关节限制
            if "range" in joint.attrib:
                range_values = [float(x) for x in joint.get("range").split()]
                if len(range_values) >= 2:
                    limit = ET.SubElement(urdf_joint, "limit")
                    limit.set("lower", str(range_values[0]))
                    limit.set("upper", str(range_values[1]))
                    
                    # 添加努力限制和速度限制
                    # 查找对应的motor
                    motor = mujoco_root.find(f".//actuator/motor[@joint='{joint_name}']")
                    if motor is not None and "ctrlrange" in motor.attrib:
                        ctrl_range = [float(x) for x in motor.get("ctrlrange").split()]
                        if len(ctrl_range) >= 2:
                            limit.set("effort", str(max(abs(ctrl_range[0]), abs(ctrl_range[1]))))
                    
                    # 默认速度限制
                    limit.set("velocity", "10")  # 默认值
            
            # 处理关节阻尼
            if "damping" in joint.attrib:
                damping = float(joint.get("damping"))
                dynamics = ET.SubElement(urdf_joint, "dynamics")
                dynamics.set("damping", str(damping))
                
                # 如果有摩擦参数
                if "frictionloss" in joint.attrib:
                    friction = float(joint.get("frictionloss"))
                    dynamics.set("friction", str(friction))
            
            joints[joint_name] = urdf_joint
        
        # 递归处理子体
        for child_body in mujoco_body.findall("body"):
            process_body(child_body, body_name)
    
    # 处理所有的body
    for body in world_body.findall("body"):
        process_body(body)
    
    # 处理传感器和执行器 (可以添加到URDF的<gazebo>标签或ROS控制配置中)
    # 但这超出了基本URDF的范围
    
    # 将URDF树写入文件
    urdf_tree = ET.ElementTree(urdf_root)
    
    # 使用自定义的indent函数代替ET.indent
    indent(urdf_root)
    
    urdf_tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"已将MuJoCo模型转换为URDF并保存至: {urdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MuJoCo XML to URDF")
    parser.add_argument("mujoco_path", help="Path to the MuJoCo XML file")
    parser.add_argument("urdf_path", help="Path to save the URDF file")
    parser.add_argument("--mesh_dir", default="meshes", help="Directory for meshes (default: meshes)")
    
    args = parser.parse_args()
    
    convert_mujoco_to_urdf(args.mujoco_path, args.urdf_path, args.mesh_dir)