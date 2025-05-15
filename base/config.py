'''
**************************************************************************

* @file         config.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.1.0"
* @brief        base config

"*************************************************************************
'''

import numpy as np
import os
import yaml
import torch

class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        config_dir = os.path.dirname(os.path.abspath(file_path))
        for key, val in cfg.items():
            if "path" in key.lower():
                if isinstance(val, str):
                    if not os.path.isabs(val):
                        cfg[key] = os.path.abspath(os.path.join(config_dir, val))
                elif isinstance(val, list):
                    new_list = []
                    for v in val:
                        if isinstance(v, str) and not os.path.isabs(v):
                            new_list.append(os.path.abspath(os.path.join(config_dir, v)))
                        else:
                            new_list.append(v)
                    cfg[key] = new_list

        device = cfg.get("device", None)
        if device not in ["cpu", "cuda:0"]:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cfg["device"] = device
        cfg["torch_device"] = torch.device(device)
        cfg["dft_dof_pos"] = np.array([cfg["dft_dof_pos"][name] for name in cfg["robot_joint"]])

        self.cfg = cfg

    def __getitem__(self, key):
        if key in self.cfg:
            return self.cfg[key]
        else:
            print(f"Warning: The key '{key}' was not found in the configuration.")
            return None

    def get(self, key, default=None):
        return self.cfg.get(key, default)
