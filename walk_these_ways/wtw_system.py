'''
**************************************************************************

* @file         him_deploy.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.0.0"
* @brief        him_deploy

"*************************************************************************
'''

import cv2
import sys
import time
import math
import torch
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

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
            lambda lx: {"y":  - 0.6 * lx},
            lambda lx: {"stance_width": 0.275 + 0.175 * lx}
        ]

        self.right_funcs = [
            lambda ry: {"freq":(1 + ry) / 2 * (cfg['cmd_freq_max'] - cfg['cmd_freq_min']) + cfg['cmd_freq_min'] },
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

        self.depth_time = time.time()
        self.first_depth = True
        self.depth_size = cfg["depth_dst_cols"] * cfg["depth_dst_rows"]
        self.depth_his_size = cfg["his_depth_num"] * self.depth_size 
        self.his_depth_obs = torch.zeros(self.depth_his_size, 
                                         dtype=torch.float,
                                         device= cfg["device"], 
                                         requires_grad=False)

        self.last_vxyt = np.array([1.0, 0.0, 0.0])
        self.new_depth = False
        self.turn_flag = False
        self.sum_dis = 0.0
        self.last_t = 0.0

        self.enable_obt = False

    def load_policy(self):
        cfg = self.cfg
        device = cfg["torch_device"]

        adap_model = torch.jit.load(cfg["model_path"][0], map_location=device)
        body_model = torch.jit.load(cfg["model_path"][1], map_location=device)

        def policy(obs):
            obs_hist = obs["his_obs"].unsqueeze(0)  # (1, N)
            #torch.save(obs_hist, "/home/nhy/Aliengo/aliengo_deploy/walk_these_ways/tensor.pt")
            #obs_hist = torch.load("/home/nhy/Aliengo/aliengo_deploy/walk_these_ways/tensor.pt")
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
                self.enable_dect()

                #if 1:
                if self.enable_obt:
                    self.avoid_obstacles(obs)

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
        grav = self.robot.get_grav()
        dof_pos = self.robot.get_dof_pos()
        dof_vel = self.robot.get_dof_vel()
        act = self.act
        last_act = self.last_act
        clock = self.get_robot_clock()
        
        dof_pos -= cfg["dft_dof_pos"]
        dof_pos  = dof_pos[cfg["joint_idx_rob2pol"]]
        dof_vel  = dof_vel[cfg["joint_idx_rob2pol"]]
        cmd     *= cfg["cmd_scale"]
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
        self.his_obs[-cfg["obs_size"]:] = obs_tensor.clone()

        cur_time = time.time()
        if (cur_time - self.depth_time > cfg["depth_dt"]):
            self.depth = self.robot.get_depth()
            self.depth_time = cur_time
            self.new_depth = True

            # depth_tensor = torch.from_numpy(depth.astype(np.float32)).float().flatten().to(cfg["torch_device"])
            # depth_tensor = torch.load("/home/nhy/Aliengo/aliengo_deploy/HIMLoco/data/depth.pt", map_location=cfg["torch_device"])
            # depth_tensor = depth_tensor.flatten()
            
            # depth_tensor[depth_tensor == -np.inf] = -6
            # depth_tensor = depth_tensor.div_(1000)
            # depth_tensor[depth_tensor < -6] = -6
            # depth_tensor.div_(6).add_(1)

            # if self.first_depth:
            #     self.first_depth = False
            #     self.his_depth_obs[:] = depth_tensor.repeat(cfg["his_depth_num"])
            # else:
            #     self.his_depth_obs[:-self.depth_size] = self.his_depth_obs[self.depth_size:].clone()
            #     self.his_depth_obs[-self.depth_size:] = depth_tensor 

        return {'his_obs': self.his_obs, 'depth': self.depth}

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
            cmd["x"] * self.cfg["ctrl_x_scale"], cmd["y"] * self.cfg["ctrl_y_scale"], 
            cmd["yaw"] * self.cfg["ctrl_yaw_scale"], cmd["height"], cmd["freq"],
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
        jot_pos = self.cfg["dft_dof_pos"] + act[self.cfg["joint_idx_pol2rob"]]
        
        return jot_pos

    def check_robot_sts(self):
        rpy_thd = 0.8
        rpy = self.robot.get_rpy()
        if max(abs(rpy[0]), abs(rpy[1])) > rpy_thd:
            print("[WARNING] Robot rpy out of range!")
            return False
        return True

    def avoid_obstacles(self, obs):

        cfg = self.cfg
        vx, vy, vt = 0.7, 0.0, 0.0
        obs_tensor = obs["his_obs"][-cfg["obs_size"]:]
        obs_tensor[3:6] = torch.tensor([vx, vy, vt], 
                                       dtype=obs_tensor.dtype, 
                                       device=cfg["torch_device"])

        if 'depth' not in obs:
            return obs
        
        if not self.new_depth:
            obs_tensor[3:6] = torch.tensor(self.last_vxyt, 
                                           dtype=obs_tensor.dtype, 
                                           device=cfg["torch_device"])
            return obs
        
        depth = np.copy(obs['depth'])
        self.new_depth = False

        # 显示深度图
        depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        depth_color = np.flip(depth_color, axis=(0, 1))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_color)
        cv2.waitKey(1)

        depth = cv2.resize(
            depth, 
            (cfg["depth_dst_cols"], cfg["depth_dst_rows"]), 
            interpolation=cv2.INTER_LINEAR
        )

        clip_cols = 2
        depth = depth[:, clip_cols:-clip_cols]
        depth = depth / 1000.0
        depth[(depth<0.25) | (depth>6.0)] = 6.0
        depth = cv2.GaussianBlur(depth, (3,3), 1.5)
        depth = np.flip(depth, axis=(0, 1))
        self.new_depth = False

        zoom_factor = 16
        depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_zoomed = cv2.resize(depth_display, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Depth", depth_zoomed)
        cv2.waitKey(1)

        # 相机参数
        K = self.robot.cam.get_depth_intrinsics()
        col_scale = cfg["depth_dst_cols"] / cfg["depth_cam_cols"]
        row_scale = cfg["depth_dst_rows"] / cfg["depth_cam_rows"]
        fx, cx = K[0][0] * col_scale, K[0][2] * col_scale - clip_cols
        fy, cy = K[1][1] * row_scale, K[1][2] * row_scale - clip_cols

        # 深度图转点云
        h, w = depth.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # 重力方向旋转对齐
        grav = obs_tensor[0:3].cpu().numpy()
        grav /= np.linalg.norm(grav)
        dst_grav = np.array([0, 0, -1.0])
        axis = np.cross(dst_grav, grav)
        angle = np.arccos(np.clip(np.dot(dst_grav, grav), -1.0, 1.0))
        if np.linalg.norm(axis) > 1e-6:
            rotation = R.from_rotvec(axis / np.linalg.norm(axis) * angle)
            points = rotation.apply(points)

        # 记录原始点云
        o3d.io.write_point_cloud("data/cloud.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))

        # 参数设置
        vt_thd = 0.15
        turn_dis_thd = 1.0
        stop_dis_thd = 0.5
        pass_width = 0.2
        cell_scale = 0.05
        pass_num = int(pass_width / cell_scale)
        view_num = 20
        view_scale = 0.5

        l_obt, r_obt = np.full(pass_num, 6.0), np.full(pass_num, 6.0)
        l_buff, r_buff = np.full(view_num, turn_dis_thd), np.full(view_num, turn_dis_thd)

        # 遍历点云计算
        filtered_pts = []
        for pt in points:
            x, y, z = pt
            if y < -0.3 or y > 0.45:
                continue
            filtered_pts.append(pt)

            angle_deg = np.degrees(np.arctan2(x, z))
            idx_view = int(angle_deg * view_scale)
            if abs(idx_view) >= view_num:
                continue

            dist = math.sqrt(pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2])
            dist = max(0.3, min(dist, turn_dis_thd))
            if angle_deg <= 0:
                l_buff[-idx_view] = min(l_buff[-idx_view], dist)
            else:
                r_buff[idx_view] = min(r_buff[idx_view], dist)

            idx_pass = int(abs(x) / cell_scale)
            if idx_pass >= pass_num:
                continue

            if x <= 0:
                l_obt[idx_pass] = min(l_obt[idx_pass], z)
            else:
                r_obt[idx_pass] = min(r_obt[idx_pass], z)

        # 记录筛选后点云
        o3d.io.write_point_cloud("data/cloud1.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered_pts)))

        # 判断通行情况
        pass_obt = np.concatenate((l_obt, r_obt))
        print(pass_obt)
        pass_obt_dis = pass_obt[np.argmin(pass_obt)]
        l_dis = np.mean(l_buff)
        r_dis = np.mean(r_buff)

        # 决策逻辑
        print(r_dis, l_dis)
        if r_dis < turn_dis_thd and l_dis < turn_dis_thd:
            vx = 0.0
            vt = vt_thd
        else:
            if pass_obt_dis < turn_dis_thd:
                v_ratio = (pass_obt_dis - stop_dis_thd) / (turn_dis_thd - stop_dis_thd)
                vx *= v_ratio
                vt = -vt_thd if r_dis > l_dis else vt_thd




        self.last_vxyt = [vx, vy, vt]
        obs_tensor[3:6] = torch.tensor(self.last_vxyt, dtype=obs_tensor.dtype, device=cfg["torch_device"])
        
        return obs

    def enable_dect(self):
        robot = self.robot
        if robot.get_ctr_state("down_psd"):
            self.enable_obt = not self.enable_obt
            robot.set_ctr_state("down_psd", False)
            
