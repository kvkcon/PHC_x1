from mjcf_urdf_simple_converter import convert

convert("x1.xml", "x1_awr.urdf", asset_file_prefix="package://x1_urdf/armwalker_mujoco_xml/meshes/")
convert("x1.xml", "x1_aw.urdf")