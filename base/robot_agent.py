'''
**************************************************************************

* @file         robot_agent.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.1.0"
* @brief        robot agent

"*************************************************************************
'''
import sys
import time
import threading
import numpy as np

import robot_interface as sdk
from utils import helpers
from base.controller import RCController
from base.realsense import RealSenseCamera
from base.livox_lidar import LivoxLidar

class RobotAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cmd_flag = False

        if cfg["controller_enable"]:
            self.ctr = RCController()

        if cfg["depth_enable"]:
            self.cam = RealSenseCamera()
            self.cam.start_depth_stream(
                cfg["depth_cam_cols"], 
                cfg["depth_cam_rows"], 
                cfg["depth_cam_fps"]
            )

        if cfg["lidar_enable"]:
            self.lid = LivoxLidar(cfg)

        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()

        self.rpy = np.zeros(3)
        self.smt_rati = 0.2
        self.smt_len = 12
        self.dt_buff = np.zeros((self.smt_len, 1))
        self.rpy_buff = np.zeros((self.smt_len, 3))

        self.cmt_thd = threading.Thread(target=self.robot_cmt)
        self.cmt_thd.start()

    def robot_cmt(self):
        LOWLEVEL = 0xff
        TARGET_PORT = 8007
        LOCAL_PORT = 8082
        TARGET_IP = "192.168.123.10"   # target IP address

        LOW_CMD_LENGTH = 610
        LOW_STATE_LENGTH = 771

        udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
        safe = sdk.Safety(sdk.LeggedType.Aliengo)
            
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        udp.InitCmdData(self.cmd)
        self.cmd.levelFlag = LOWLEVEL

        udp.SetSend(self.cmd)
        udp.Send()

        buff_idx = 0
        last_rpy = np.zeros(3)
        last_time = time.time()

        while True:
            try:
                time.sleep(self.cfg["control_dt"])
                udp.Recv()
                udp.GetRecv(self.state)

                if self.cfg["controller_enable"]:
                    self.ctr.update(self.state.wirelessRemote)

                rpy = np.array(self.state.imu.rpy)
                cur_time = time.time()
                self.rpy_buff[buff_idx] = rpy - last_rpy

                self.dt_buff[buff_idx] = cur_time - last_time

                last_rpy, last_time = rpy, cur_time
                buff_idx = (buff_idx + 1) % self.smt_len

                if self.cmd_flag:
                    safe.PositionLimit(self.cmd)
                    safe.PowerProtect(self.cmd, self.state, 9)
                    udp.SetSend(self.cmd)
                    udp.Send()

            except Exception as e:
                print(f"[ERROR] robot_cmt Exception: {e}")

    def set_tar_dof_pos(self, tar_dof_pos):
        self.cmd_flag = True
        for i in range(len(self.cfg["robot_joint"])):
            self.cmd.motorCmd[i].q = tar_dof_pos[i]
            self.cmd.motorCmd[i].dq  = 0.0
            self.cmd.motorCmd[i].tau = 0.0
            self.cmd.motorCmd[i].Kp  = self.cfg["kp"]
            self.cmd.motorCmd[i].Kd  = self.cfg["kd"]

    def set_motor_cmd(self, motor_cmd):
        self.cmd_flag = True
        q = motor_cmd["q"]
        dq = motor_cmd["dq"]
        tau = motor_cmd["tau"]
        Kp = motor_cmd["Kp"]
        Kd = motor_cmd["Kd"]

        for i in range(len(self.cfg["robot_joint"])):
            self.cmd.motorCmd[i].q = q[i]
            self.cmd.motorCmd[i].dq  = dq[i]
            self.cmd.motorCmd[i].tau = tau[i]
            self.cmd.motorCmd[i].Kp  = Kp[i]
            self.cmd.motorCmd[i].Kd  = Kd[i]

    def get_tar_dof_pos(self):
        return np.array(
            [self.cmd.motorCmd[i].q for i in range(len(self.cfg["robot_joint"]))],
            dtype=np.float32
        )

    def get_grav(self):
        R = helpers.rpy2SO3(self.state.imu.rpy)
        return np.dot(R.T, np.array([0, 0, -1]))

    def get_rpy(self):
        return self.state.imu.rpy

    def get_gyr(self):
        _gyr = np.mean(self.rpy_buff / self.dt_buff, axis=0)
        return self.smt_rati * _gyr + (1 - self.smt_rati) * self.rpy

    def get_dof_pos(self):
        return np.array(
            [self.state.motorState[i].q for i in range(len(self.cfg["robot_joint"]))],
            dtype=np.float32
        )

    def get_dof_vel(self):
        return np.array(
            [self.state.motorState[i].dq for i in range(len(self.cfg["robot_joint"]))],
            dtype=np.float32
        )

    def get_remote(self):
        return self.state.wirelessRemote

    def get_ctr_state(self, name):
        if not self.cfg["controller_enable"]:
            print("Error: controller is not enabled, checke config.", file=sys.stderr)
            sys.exit(1)

        if name not in self.ctr.data:
            print(f"Error: unexpected button '{name}'.", file=sys.stderr)
            sys.exit(1)

        return self.ctr.data.get(name, False)

    def set_ctr_state(self, name, value):
        if not self.cfg["controller_enable"]:
            print("Error: controller is not enabled, checke config.", file=sys.stderr)
            sys.exit(1)

        if name not in self.ctr.data:
            print(f"Error: unexpected button '{name}'.", file=sys.stderr)
            sys.exit(1)
        self.ctr.data[name] = value

    def get_depth(self):
        if not self.cfg["depth_enable"]:
            print("Error: depth is not enabled, checke config.", file=sys.stderr)
            sys.exit(1)

        depth = self.cam.get_depth()
        return depth
