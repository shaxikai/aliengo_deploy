'''
**************************************************************************

* @file         system.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.1.0"
* @brief        base system

"*************************************************************************
'''

import time
import copy
import torch
import numpy as np

from base.config import Config
from base.robot_agent import RobotAgent

class System:
    def __init__(self):
        self.robot  = RobotAgent(self.cfg)
        self.policy = self.load_policy()

    def load_policy(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def robot_reset(self):
        robot = self.robot
        joint_pos = robot.get_dof_pos()
        dft_dof_pos = self.cfg["dft_dof_pos"]

        act_clip = 0.05
        act_seq = []
        act = joint_pos - dft_dof_pos
        while np.max(np.abs(act)) > 0.01:
            act -= np.clip(act, -act_clip, act_clip)
            act_seq.append(copy.deepcopy(act)) 

        for act in act_seq:
            robot.set_tar_dof_pos(act)
            time.sleep(0.05)

    def get_robot_obs(self):
        raise NotImplementedError

    def get_robot_cmd(self):
        raise NotImplementedError


