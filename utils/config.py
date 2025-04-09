'''
**************************************************************************

* @file         Config.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.0.0"
* @brief        robot Config

"*************************************************************************
'''

from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml
import torch

class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            
            device = cfg.get("device", None)
            if device not in ["cpu", "cuda:0"]:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"

            cfg["device"] = device
            cfg["torch_device"] = torch.device(device)

            cfg["dft_dof_pos"] = np.array([cfg["dft_dof_pos"][name] for name in cfg["robot_joint"]])

            cfg["joint_idx_rob2pol"] = [cfg["policy_joint"].index(name) for name in cfg["robot_joint"]]
            cfg["joint_idx_pol2rob"] = [cfg["robot_joint"].index(name) for name in cfg["policy_joint"]]

            self.cfg = cfg

    def __getitem__(self, key):
        if key in self.cfg:
            return self.cfg[key]
        else:
            print(f"Warning: The key '{key}' was not found in the configuration.")
            return None

