'''
**************************************************************************

* @file         wtw_config.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.1.0"
* @brief        wtw_config

"*************************************************************************
'''

import numpy as np
from base.config import Config

class WTWConfig(Config):

    def __init__(self, file_path):
        super().__init__(file_path)

        cfg = self.cfg
        cfg["joint_idx_pol2rob"] = [cfg["policy_joint"].index(name) for name in cfg["robot_joint"]]
        cfg["joint_idx_rob2pol"] = [cfg["robot_joint"].index(name) for name in cfg["policy_joint"]]
        self.cfg = cfg

        keys = [
            'cmd_x_scale', 'cmd_y_scale', 'cmd_yaw_scale', 'cmd_height_scale','cmd_freq_scale', 
            'cmd_phase_scale', 'cmd_phase_scale', 'cmd_phase_scale', 'cmd_phase_scale',
            'cmd_footswing_scale', 'cmd_pitch_scale', 'cmd_roll_scale', 'cmd_stand_width_scale', 
            'cmd_stand_length_scale', 'cmd_aux_reward_scale'
        ]

        cfg["cmd_scale"] = np.array([cfg[key] for key in keys])


    

