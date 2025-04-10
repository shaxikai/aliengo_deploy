'''
**************************************************************************

* @file         wtw_config.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.0.0"
* @brief        wtw_config

"*************************************************************************
'''

from base.config import Config

class WTWConfig(Config):

    def __init__(self, file_path):
        super().__init__(file_path)

        cfg = self.cfg
        cfg["joint_idx_rob2pol"] = [cfg["policy_joint"].index(name) for name in cfg["robot_joint"]]
        cfg["joint_idx_pol2rob"] = [cfg["robot_joint"].index(name) for name in cfg["policy_joint"]]
        self.cfg = cfg


