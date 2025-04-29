'''
**************************************************************************

* @file         him_deploy.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.0.0"
* @brief        him_deploy

"*************************************************************************
'''

import sys
import time
import torch
import numpy as np

from base.system import System
from HIMLoco.him_config import HIMConfig

from base.robot_agent import RobotAgent

class HIMSystem(System):
    def __init__(self, cfg_pn):
        cfg = HIMConfig(cfg_pn)
        self.cfg = cfg
        super().__init__()

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

    def robot_reset(self, max_time: float = 5.0, control_freq: float = 20.0) -> bool:
        """将机器人关节位置渐进复位到默认位置。
        
        Args:
            max_time: 最大复位时间（秒），超时则终止。
            control_freq: 控制频率（Hz），影响平滑度。
        Returns:
            bool: 是否成功复位（False表示超时或异常）。
        """
        # 1. 初始化参数
        robot = self.robot
        joint_pos = robot.get_dof_pos()
        dft_dof_pos = np.asarray(self.cfg["dft_dof_pos"])
        dof_num = len(joint_pos)
        
        # 2. 计算复位参数
        act_clip = 0.1  # 单步最大调整幅度（弧度）
        act = joint_pos - dft_dof_pos
        act_max = np.max(np.abs(act))
        num_steps = max(1, int(np.ceil(act_max / act_clip)))  # 至少1步
        
        # 3. 预分配控制指令（避免循环内重复创建）
        motor_cmd = {
            "q": np.zeros(dof_num),
            "dq": np.zeros(dof_num),
            "tau": np.zeros(dof_num),
            "Kp": np.full(dof_num, 60.0),
            "Kd": np.full(dof_num, 2.0)
        }
        
        # 4. 渐进复位
        start_time = time.time()
        for step in range(num_steps):
            # 超时检查
            if time.time() - start_time > max_time:
                print(f"RESET FAILED")
                print(f"Reset timeout after {max_time}s")
                sys.exit(1)
            
            # 计算插值位置
            ratio = (step + 1) / num_steps
            motor_cmd["q"] = joint_pos * (1 - ratio) + dft_dof_pos * ratio
            
            # 发送指令
            robot.set_motor_cmd(motor_cmd)
            time.sleep(1.0 / control_freq)
        
        # 5. 最终确认
        # final_error = np.max(np.abs(robot.get_dof_pos() - dft_dof_pos))
        # if final_error > 0.05:
        #     print(f"RESET FAILED")
        #     print(f"Max error: {final_error:.4f} rad (threshold: 0.05 rad)")
        #     sys.exit(1)

    def get_robot_obs(self):

        cfg = self.cfg
        cmd = self.get_robot_cmd()
        gyr = self.robot.get_gyr()
        grav = self.robot.get_grav()
        dof_pos = self.robot.get_dof_pos()
        dof_vel = self.robot.get_dof_vel()
        act = self.act

        cmd[:2] *= cfg["vel_scale"]
        cmd[2]  *= cfg["gyr_scale"]
        gyr     *= cfg["gyr_scale"]
        dof_pos -= cfg["dft_dof_pos"]
        dof_pos  = dof_pos[cfg["joint_idx_rob2pol"]]
        dof_vel  = dof_vel[cfg["joint_idx_rob2pol"]]
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
            depth = self.robot.get_depth()
            self.depth_time = cur_time

            depth_tensor = torch.from_numpy(depth.astype(np.float32)).float().flatten().to(cfg["torch_device"])
            # depth_tensor = torch.load("/home/nhy/Aliengo/aliengo_deploy/HIMLoco/data/depth.pt", map_location=cfg["torch_device"])
            depth_tensor = depth_tensor.flatten()
            
            depth_tensor = depth_tensor.div_(1000)
            depth_tensor[depth_tensor == -np.inf] = -6
            depth_tensor[depth_tensor < -6] = -6
            depth_tensor.div_(6).add_(1)

            if self.first_depth:
                self.first_depth = False
                self.his_depth_obs[:] = depth_tensor.repeat(cfg["his_depth_num"])
            else:
                self.his_depth_obs[self.depth_size:] = self.his_depth_obs[:-self.depth_size].clone()
                self.his_depth_obs[:self.depth_size] = depth_tensor 

        return {'his_obs': self.his_obs, 'his_depth_obs': self.his_depth_obs}

    def get_robot_cmd(self):
        cmd = np.array([0.0, 0.0, 0.0])
        return cmd
    
    def act2joint(self, act):
        act = act.detach().cpu().numpy()
        act = np.clip(act, -self.cfg["act_clip"], self.cfg["act_clip"])
        self.act = act.copy()

        act *= self.cfg["act_scale"]
        act[[0, 3, 6, 9]] *= self.cfg["hip_reduction"]
        jot_pos = self.cfg["dft_dof_pos"] + act[self.cfg["joint_idx_pol2rob"]]
        return jot_pos

    def check_robot_sts(self):
        rpy_thd = 1.6
        rpy = self.robot.get_rpy()
        if max(abs(rpy[0]), abs(rpy[1])) > rpy_thd:
            print("[WARNING] Robot rpy out of range!")
            return False
        return True

