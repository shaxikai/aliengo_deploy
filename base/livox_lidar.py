'''
**************************************************************************

* @file         livox_lidar.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.1.0"
* @brief        robot lidar using livox mid360

"*************************************************************************
'''

import sys
import threading
import numpy as np
import livox_sdk

class LivoxLidar:
    def __init__(self, cfg):

        self.cfg = cfg
        self.pts_buff = []
        self.pts = []
        self.last_ts = None
        self.frame_id = 0

        self.imu = {
            "ts": np.uint64(0),
            "gyr": np.zeros(3, dtype=np.float32),
            "acc": np.zeros(3, dtype=np.float32)
        }

        self.pts_cb = None
        self.imu_cb = None

        self.lock_pts = threading.Lock()
        self.lock_imu = threading.Lock()

        lid_cfg = livox_sdk.PyLivoxLidarCfg()
        for field in [
                "lidar_name", "lidar_ipaddr", "host_ip", "multicast_ip",
                "lidar_subnet_mask", "lidar_gateway", "lidar_cmd_data_port", 
                "lidar_push_msg_port",  "lidar_point_data_port",  "lidar_imu_data_port", 
                "lidar_log_data_port",  "host_cmd_data_port",     "host_push_msg_port", 
                "host_point_data_port", "host_imu_data_port",     "host_log_data_port"
            ]:
            setattr(lid_cfg, field, cfg.get(field, ""))

        if not livox_sdk.LivoxLidarSdkInitFromCfg(lid_cfg):
            print("ERROR: Lidar Init Failure.")
            sys.exit(1)

        livox_sdk.SetLivoxLidarPointCloudCallBack(self.livox_sdk_pointcloud_callback)
        livox_sdk.SetLivoxLidarImuDataCallback(self.livox_sdk_imu_callback)
        
    def livox_sdk_pointcloud_callback(self, packet):
        raw = packet.get_pts_data()
        if not raw:
            print("WARRING: No Lidar Data.")
            return

        dtype = np.dtype([
            ('x', 'i4'),
            ('y', 'i4'),
            ('z', 'i4'),
            ('reflectivity', 'u1'),
            ('tag', 'u1'),
        ])
        pts = np.frombuffer(raw, dtype=dtype)
        if len(pts) == 0:
            print("WARRING: Lidar Data No Pts.")
            return

        pts = np.stack([pts['x'], pts['y'], pts['z']], axis=-1) * 0.001

        cur_ts = packet.ts  
        if self.last_ts is None:
            self.last_ts = cur_ts

        self.pts_buff.append(pts)

        if cur_ts - self.last_ts >= self.cfg["lidar_frame_dur"] * 1e9:

            with self.lock_pts:
                self.pts = np.concatenate(self.pts_buff, axis=0)

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(all_points)
            # filename = f"data/frame.pcd"
            # o3d.io.write_point_cloud(filename, pcd)
            # print(f"[保存] 写入 {filename}，包含 {len(all_points)} 个点")

            if self.pts_cb:
                self.pts_cb(self.pts.copy())

            self.frame_id += 1
            self.last_ts = None
            self.pts_buff = []

    def livox_sdk_imu_callback(self, packet):
        self.imu["ts"]  = packet.ts
        self.imu["gyr"] = np.array(packet.gyr, dtype=np.float32)
        self.imu["acc"] = np.array(packet.acc, dtype=np.float32)
        if self.imu_cb:
            self.imu_cb(self.imu.copy()) 

    def get_pts_data(self):
        with self.lock_pts:
            return self.pts.copy()

    def get_imu_data(self):
        with self.lock_imu:
            return self.imu.copy()

    def set_pts_cb(self, cb):
        self.pts_cb = cb

    def set_imu_cb(self, cb):
        self.imu_cb = cb

    def __del__(self):
        livox_sdk.LivoxLidarSdkUninit()
