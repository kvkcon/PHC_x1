from mjcf_urdf_simple_converter import convert

convert("phc/data/assets/robot/zhiyuan_x1/x1.xml", "phc/data/assets/robot/zhiyuan_x1/urdf/x1_awr.urdf", asset_file_prefix="package://x1_urdf/armwalker_mujoco_xml/meshes/")
convert("phc/data/assets/robot/zhiyuan_x1/x1.xml", "phc/data/assets/robot/zhiyuan_x1/urdf/x1_aw.urdf")