## Motion Retargeting  - >  PHC
---

`PHC` Retargeting         *test on `AMASS` and `TRUMANS` dataset* 
### Overview
- Adjust `foot_tip in learning shape fitting` to solve rk errors
- Add dof_weight to make ankle roll smooth
- Clean phc codebase to extract functions into motion_retargeting


### Quick Start
#### Install
To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:


```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 

3. Place *SMPL_NEUTRAL.pkl* & *SMPLX_NEUTRAL.pkl* in `data/smpl/`

#### Retargeting
replace your **amass_root** & **motion_name** in `phc/data/cfg/config.yaml`
1. learn weight for shape
```python
python scripts/fit_process/fit_smpl_shape.py robot=unitree_g1_fitting
```
2. fit motion per frame
```python
python scripts/fit_process/fit_smpl_motion.py robot=unitree_g1_fitting
```
> robot: unitree_g1_fitting

> +motion_name=0-Transitions_mocap_mazen_c3d_dance_stand_poses

3. visualize motion
```python
python scripts/vis/vis_q_mj.py robot=unitree_g1_fitting
```
#### Tool
- convert trumans data to amass format
    ```python
    python scripts/data_pre_process/convert_trumans2amass.py data/trumans/2023-01-17@00-46-07_smplx_results.pkl data/trumans/2023-01-17@00-46-07_smplx_results.npz
    ```
- `python scripts/vis/npz_vis.py data/trumans/2023-01-17@00-46-07_smplx_results.npz`
- `python scripts/vis/pkl_vis.py data/trumans/2023-01-17@00-46-07_smplx_results.pkl`