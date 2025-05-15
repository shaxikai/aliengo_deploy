'''
**************************************************************************

* @file         realsense.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.1
* @version      V1.1.0"
* @brief        realsense camera

"*************************************************************************
'''

import sys
import numpy as np
import pyrealsense2 as rs

class RealSenseCamera:
    def __init__(self):
        ctx = rs.context()
        if len(ctx.devices) == 0:
            print("ERROR: No RealSense device connected.")
            sys.exit(1)  # 退出程

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.depth_K = np.zeros((3, 3))
        self.color_K = np.zeros((3, 3))

    def start_color_stream(self, width=640, height=480, fps=30):
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(self.config)

    def start_depth_stream(self, width=640, height=480, fps=30):
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(self.config)

        # 获取深度流的内参
        depth_profile = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        # 打印内参
        self.depth_K = np.array([
            [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
            [0, depth_intrinsics.fy, depth_intrinsics.ppy],
            [0, 0, 1]
        ])

        print("depth K:")
        print(self.depth_K)
    

    def get_color(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Warning: Failed to get color frame")
            return None
        return np.asanyarray(color_frame.get_data())

    def get_depth(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            print("Warning: Failed to get depth frame")
            return None
        return np.asanyarray(depth_frame.get_data())

    def get_depth_intrinsics(self):
        return self.depth_K

    def __del__(self):
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Warning: Failed to stop RealSense pipeline - {e}")



