'''
**************************************************************************

* @file         him_deploy.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.0.0"
* @brief        him_deploy

"*************************************************************************
'''

import time
import copy
import torch
import numpy as np

from base.system import System
from walk_these_ways.wtw_config import WTWConfig

class WTWSystem(System):
    def __init__(self, cfg_pn):
        cfg = WTWConfig(cfg_pn)
        self.cfg = cfg
        super().__init__()

        self.L1_mode = 0
        self.R1_mode = 0
        self.cmd_phase = 0.5
        self.cmd_offset = 0.0
        self.cmd_bound = 0.0
        self.cmd_duration = 0.5

        self.ABXY2POBD = {
            "A":     {"phase": 0.5, "offset": 0.0, "bound": 0.0, "duration": 0.5},
            "B":     {"phase": 0.0, "offset": 0.0, "bound": 0.0, "duration": 0.5},
            "X":     {"phase": 0.0, "offset": 0.5, "bound": 0.0, "duration": 0.5},
            "Y":     {"phase": 0.0, "offset": 0.0, "bound": 0.5, "duration": 0.5},
        }

        self.left_funcs = [
            lambda lx: {"height": 0.3 * lx},
            lambda lx: {"y": 0.6 * lx},
            lambda lx: {"stance_width": 0.275 + 0.175 * lx}
        ]

        self.right_funcs = [
            lambda ry: {"freq": (1 + ry) / 2 * 2 + 2.0},
            lambda ry: {"footswing": max(0, ry) * 0.32 + 0.03},
            lambda ry: {"ori_pitch": -0.4 * ry}
        ]

        self.gait_idx = 0.0

        self.cmds = np.zeros(cfg["cmd_size"], dtype=np.float32)

        self.dof_num = len(cfg["robot_joint"])
        self.act = np.zeros(self.dof_num, dtype=np.float32)
        self.last_act = np.zeros(self.dof_num, dtype=np.float32)

        self.his_obs_size = cfg["his_obs_num"] * cfg["obs_size"]
        self.his_obs = torch.zeros(self.his_obs_size, 
                                   dtype=torch.float,
                                   device = cfg["device"], 
                                   requires_grad=False)

    def load_policy(self):
        cfg = self.cfg
        device = cfg["torch_device"]

        adap_model = torch.jit.load(cfg["model_path"][0], map_location=device)
        body_model = torch.jit.load(cfg["model_path"][1], map_location=device)

        def policy(obs):
            obs_hist = obs.unsqueeze(0)  # (1, N)
            latent = adap_model(obs_hist)
            action = body_model(torch.cat([obs_hist, latent], dim=-1))
            return action[0]

        return policy


    def run(self):
        robot = self.robot
        dt = self.cfg["policy_dt"]

        print("reset robot [Press up]")
        while not robot.get_ctr_state("up_psd"):
            time.sleep(0.01)

        self.robot_reset()
        robot.set_ctr_state("up_psd", False)

        while True:
            print("Unlock controller [Press up]")
            while not robot.get_ctr_state("up_psd"):
                time.sleep(0.01)

            robot.set_ctr_state("up_psd", False)
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

                if robot.get_ctr_state("up_psd") or not self.check_robot_sts():
                    print("up_pressed robot reset.")
                    self.robot_reset()
                    robot.set_ctr_state("up_psd", False)
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
        grav = self.robot.get_grav()
        dof_pos = self.robot.get_dof_pos()
        dof_vel = self.robot.get_dof_vel()
        act = self.act
        last_act = self.last_act
        clock = self.get_robot_clock()

        dof_pos -= cfg["dft_dof_pos"]
        cmd[:2] *= cfg["vel_scale"]
        cmd[2]  *= cfg["gyr_scale"]
        dof_pos *= cfg["dof_pos_scale"]
        dof_vel *= cfg["dof_vel_scale"]

        obs = np.concatenate([
            grav, 
            cmd, 
            dof_pos, 
            dof_vel,
            act,
            last_act,
            clock
        ], axis=-1)
        obs = np.clip(obs, -cfg["obs_clip"], cfg["obs_clip"])
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(cfg["device"])

        self.his_obs[:-cfg["obs_size"]] = self.his_obs[cfg["obs_size"]:].clone()
        self.his_obs[-cfg["obs_size"]:] = obs_tensor

        return self.his_obs

    def get_robot_cmd(self):
        robot = self.robot

        for btn, data in self.ABXY2POBD.items():
            if robot.get_ctr_state(btn) > 0:
                self.cmd_phase, self.cmd_offset = data["phase"], data["offset"]
                self.cmd_bound, self.cmd_duration = data["bound"], data["duration"]
                break

        if robot.get_ctr_state("L1_psd"):
            self.L1_mode = (self.L1_mode + 1) % 3
            robot.set_ctr_state("L1_psd", False)
        if robot.get_ctr_state("R1_psd"):
            self.R1_mode = (self.R1_mode + 1) % 3
            robot.set_ctr_state("R1_psd", False)

        ctr = robot.get_ctr_state
        lx, ly, rx, ry = ctr("lx"), ctr("ly"), ctr("rx"), ctr("ry")

        cmd = {
            "x": ly, "y": 0., "yaw": -rx, "height": 0., "freq": 3.0,
            "footswing": 0.08, "ori_pitch": 0., "ori_roll": 0.,
            "stance_width": 0.33, "stance_length": 0.40,
        }

        cmd.update(self.left_funcs[self.L1_mode](lx))
        cmd.update(self.right_funcs[self.R1_mode](ry))

        self.cmds = np.array([
            cmd["x"] * self.cfg["cmd_x_scale"], cmd["y"] * self.cfg["cmd_y_scale"], 
            cmd["yaw"] * self.cfg["cmd_yaw_scale"], cmd["height"], cmd["freq"],
            self.cmd_phase, self.cmd_offset, self.cmd_bound, self.cmd_duration,
            cmd["footswing"], cmd["ori_pitch"], cmd["ori_roll"],
            cmd["stance_width"], cmd["stance_length"], 0,
        ])

        return self.cmds

    def get_robot_clock(self):
        freq   = self.cmds[4]
        phase  = self.cmds[5]
        offset = self.cmds[6]
        bound  = self.cmds[7]

        self.gait_idx = np.remainder(self.gait_idx + self.cfg["policy_dt"] * freq, 1.0)

        foot_indices = [
            self.gait_idx + phase + offset + bound,
            self.gait_idx + offset,
            self.gait_idx + bound,
            self.gait_idx + phase
        ]

        return np.sin(2 * np.pi * np.array(foot_indices))


    def act2joint(self, act):
        act = act.detach().cpu().numpy()
        act = np.clip(act, -self.cfg["act_clip"], self.cfg["act_clip"])
        self.last_act = self.act.copy()
        self.act = act.copy()

        act *= self.cfg["act_scale"]
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

