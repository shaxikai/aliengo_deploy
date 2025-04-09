'''
**************************************************************************

* @file         deploy.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.0.0"
* @brief        run policy

"*************************************************************************
'''

import time
import copy
import torch
import numpy as np

from utils.config import Config
from utils.robot_agent import RobotAgent

class System:
    def __init__(self, cfg_pn):
    
        cfg = Config(cfg_pn)
        self.cfg = cfg
        self.robot  = RobotAgent(cfg)
        self.policy = self.load_policy()

        self.dof_num = len(cfg["robot_joint"])
        self.act = np.zeros(self.dof_num, dtype=np.float32)



        self.his_obs_size = cfg["his_obs_num"] * cfg["obs_size"]
        self.his_obs = torch.zeros(self.his_obs_size, 
                                   dtype=torch.float,
                                   device = cfg["device"], 
                                   requires_grad=False)

        self.depth_time = time.time()
        self.first_depth = True
        self.depth_size = cfg["depth_dst_cols"] * cfg["depth_dst_rows"]
        self.depth_his_size = cfg["his_depth_num"] * self.depth_size 
        self.his_depth_obs = torch.zeros(self.depth_his_size, 
                                         dtype=torch.float,
                                         device= cfg["device"], 
                                         requires_grad=False)

    def load_policy(self):
        cfg = self.cfg
        model = torch.jit.load(cfg["model_path"], map_location=cfg["torch_device"])
        model.eval()

        def policy(obs):
            act = model.forward(obs['his_obs'].unsqueeze(0), obs['his_depth_obs'].unsqueeze(0))
            return act[0]

        return policy

    def run(self):
        robot = self.robot
        dt = self.cfg["policy_dt"]

        # print("reset robot [Press R1]")
        # while not robot.get_ctr_state("R1_psd"):
        #     time.sleep(0.01)

        self.robot_reset()
        robot.set_ctr_state("R1_psd", False)

        while True:
            # print("Unlock controller [Press R1]")
            # while not robot.get_ctr_state("R1_psd"):
            #     time.sleep(0.01)

            robot.set_ctr_state("R1_psd", False)
            obs = self.get_robot_obs()

            last_time = time.time()
            while True:
                act = self.policy(obs)
                dof_pos = self.act2joint(act)
                robot.set_tar_dof_pos(dof_pos)

                cur_time = time.time()
                time.sleep(max(dt - (cur_time - last_time), 0))
                last_time = cur_time
                obs = self.get_robot_obs()

                if robot.get_ctr_state("R1_psd") or not self.check_robot_sts():
                    print("R1_pressed robot reset.")
                    self.robot_reset()
                    robot.set_ctr_state("R1_psd", False)
                    break

                if not self.check_robot_sts():
                    print("robot status error robot reset.")
                    self.robot_reset()
                    break

                if robot.get_ctr_state("R2_psd"):
                    print("R2_pressed exit.")
                    self.robot_reset()
                    robot.set_ctr_state("R2_psd", False)
                    return

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

        cfg = self.cfg
        cmd = self.get_robot_cmd()
        gyr = self.robot.get_gyr()
        grav = self.robot.get_grav()
        dof_pos = self.robot.get_dof_pos()
        dof_vel = self.robot.get_dof_vel()
        act = self.act

        dof_pos -= cfg["dft_dof_pos"]
        cmd[:2] *= cfg["vel_scale"]
        cmd[2]  *= cfg["gyr_scale"]
        gyr     *= cfg["gyr_scale"]
        dof_pos *= cfg["dof_pos_scale"]
        dof_vel *= cfg["dof_vel_scale"]

        obs = np.concatenate([
            cmd, 
            gyr, 
            grav, 
            dof_pos, 
            dof_vel,
            act
        ], axis=-1)
        obs = np.clip(obs, -cfg["obs_clip"], cfg["obs_clip"])
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(cfg["device"])

        self.his_obs[cfg["obs_size"]:] = self.his_obs[:-cfg["obs_size"]].clone()
        self.his_obs[:cfg["obs_size"]] = obs_tensor

        cur_time = time.time()
        if (cur_time - self.depth_time > cfg["depth_dt"]):
            #depth = self.robot.get_depth()
            depth = np.zeros((cfg["depth_dst_rows"], cfg["depth_dst_cols"]), dtype=np.float32)
            self.depth_time = cur_time

            depth_tensor = torch.from_numpy(depth).float().flatten().to(cfg["torch_device"])
            depth_tensor.div_(6).add_(1)

            if self.first_depth:
                self.first_depth = False
                self.his_depth_obs[:] = depth_tensor.repeat(cfg["his_depth_num"])
            else:
                self.his_depth_obs[self.depth_size:] = self.his_depth_obs[:-self.depth_size].clone()
                self.his_depth_obs[:self.depth_size] = depth_tensor 

        return {'his_obs': self.his_obs, 'his_depth_obs': self.his_depth_obs}

    def get_robot_cmd(self):
        cmd = np.array([1.0, 0.0, 0.0])
        return cmd
    
    def act2joint(self, act):
        act = act.detach().cpu().numpy()
        act = np.clip(act, -self.cfg["act_clip"], self.cfg["act_clip"])
        self.act = act.copy()

        act *= self.cfg["action_scale"]
        act[[0, 3, 6, 9]] *= self.cfg["hip_reduction"]
        jot_pos = self.cfg["dft_dof_pos"] + act[self.cfg["joint_idx_rob2pol"]]
        return jot_pos

    def check_robot_sts(self):
        rpy_thd = 1.6
        rpy = self.robot.get_rpy()
        if max(abs(rpy[0]), abs(rpy[1])) > rpy_thd:
            print("[WARNING] Robot rpy out of range!")
            return False
        return True

