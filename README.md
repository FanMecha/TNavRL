# *TNavRL: Cross-Modal Transformer for Humanoid Visual Navigation*
[![Python](https://img.shields.io/badge/python-3.10-4B8BBE.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-NVIDIA-C0392B.svg)](https://github.com/NVIDIA-Omniverse/IsaacLab)
[![Linux platform](https://img.shields.io/badge/platform-Ubuntu-27AE60.svg)](https://releases.ubuntu.com/22.04/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-ffd21e.svg)](https://github.com/huggingface/transformers)
[![Conda](https://img.shields.io/badge/Conda-%2344A833.svg?style=flat&logo=anaconda&logoColor=white)](https://anaconda.org/)

*Welcome to the TNavRL repository，which was designed to enable humanoid robots to safely navigate cluttered environments using Reinforcement Learning. And it can be extended to any robot that adopts a velocity-based control system.
For additional details, please refer to the related paper available as below:*

*Fan Huang; HaiMing Mou; QingDu Li. "TNavRL: Cross-Modal Transformer for Humanoid Visual Navigation" IEEE Robotics and Automation Letters (RA-L), 2026.*

[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://ieeexplore.ieee.org/abstract/document/11419774)
[![Bilibili](https://img.shields.io/badge/Bilibili-FF2442)](https://b23.tv/wEGkaip)
[![小红书](https://img.shields.io/badge/小红书-FF2442)](https://xhslink.com/m/67jsES7t32)

## 🔨 *Install*
#### *install isaacsim*
####
    * conda create -n TNavRL python=3.10
    * conda activate TNavRL 
    * pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    * pip install --upgrade pip
    * pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
    * isaacsim (verify install)

#### *install isaaclab*
#### 
    * cd TNavRL
    * sudo apt install cmake build-essential
    * cd rsl_rl
    * pip install -e .
    * cd ..
    * ./isaaclab.sh --install
    * ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py (verify install)

## 📌 *Training*
*If you want to train the navigation policy of your robot, you must add the robot assets and training the locomotion policy of your robot first. And you should change the directory with your own content*
#### *change your robot model*
    * add your robot assets in the directory: TNavRL/source/isaaclab_assets/isaaclab_assets/robots
    * import your robot assets in the dircetory: TNavRL/source/isaaclab_assets/isaaclab_assets/robots/__init__.py

#### *train the locomotion policy of your robot*
    * create your robot locomotion config in the dircetory: TNavRL/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
    * add the code "from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import *" in the dircetory: TNavRL/source/isaaclab/isaaclab/envs/mdp/__init__.py

#### *train navigation*
####
    * create your robot navigation config in the dircetory: TNavRL/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/config/
    * import your navigation config in the dircetory: TNavRL/source/isaaclab_tasks/isaaclab_tasks/manager_based/navigation/__init__.py
    * ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=X2-Navigation-Train --num_envs=4096 --max_iterations=1000 --headless --logger=wandb
*If you want to train our robot navigation policy, we already provide the pretrained locomotion policy. So you can directly train the navigation policy of our robot with the instruction*

*!Tips: We train the navigation policy in 48 GPU Nvidia 4090D. If you meet the OOM mistake, you should reduce num_envs as like 2048 or 1024?*

## 🤝 *Contribution*
*Contributions, discussions, and ideas are welcome. If our work is useful to your research, please consider citing our paper and giving it a ⭐️!*
####
    @ARTICLE{11419774,
    author={Huang, Fan and Mou, HaiMing and Li, QingDu},
    journal={IEEE Robotics and Automation Letters}, 
    title={TNavRL: Cross-Modal Transformer for Humanoid Visual Navigation}, 
    year={2026},
    volume={11},
    number={5},
    pages={5374-5381},
    keywords={Navigation;Humanoid robots;Propioception;Visualization;Robots;Training;Transformers;Pipelines;Collision avoidance;Feature extraction;Visual navigation;cross-modal transformer;reinforcement learning;humanoid robot},
    doi={10.1109/LRA.2026.3669788}}
*Based on our previous work, we have introduced [LSTM-SRU](https://journals.sagepub.com/doi/full/10.1177/02783649251401926) as a spatial memory module, following recent advances in the literature, to enhance long-term environment understanding and navigation performance. Thanks their great work!*