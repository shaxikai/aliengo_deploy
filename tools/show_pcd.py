'''
**************************************************************************

* @file         show_pcd.py
* @author       Wei Wang -> shaxikai@outlook.com
* @date         2025.4.23
* @version      V1.1.0"
* @brief        show point cloud

"*************************************************************************
'''


import open3d as o3d
import sys
import os
import numpy as np

def main():
    # 可修改为你的点云文件路径，例如 'cloud.ply'、'cloud.pcd'
    file_path = "data/cloud.ply"

    if not os.path.exists(file_path):
        print(f"点云文件不存在: {file_path}")
        sys.exit(1)

    # 读取点云
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"已加载点云: {file_path}")
    print(pcd)

    # 创建坐标系可视化
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # 定义线段的起点和终点（3D坐标）
    hw, l, d, u= 0.2, 6.0, 0.45, -0.3
    points = np.array([
        [-hw,   d, 0],  # 第0个点
        [ hw,   d, 0],  # 第1个点
        [ hw,   u, 0],  # 第2个点
        [-hw,   u, 0],   # 第3个点
        [-hw,   d, l],  # 第0个点
        [ hw,   d, l],  # 第1个点
        [ hw,   u, l],  # 第2个点
        [-hw,   u, l]   # 第3个点
    ])

    lines = np.array([
        [0, 1],  # 连接第0个点和第1个点
        [1, 2],  # 连接第1个点和第2个点
        [2, 3],  # 连接第2个点和第3个点
        [3, 0],   # 连接第3个点和第0个点
        [4, 5],  # 连接第0个点和第1个点
        [5, 6],  # 连接第1个点和第2个点
        [6, 7],  # 连接第2个点和第3个点
        [7, 4],   # 连接第3个点和第0个点
        [0, 4],  # 连接第0个点和第1个点
        [1, 5],  # 连接第1个点和第2个点
        [2, 6],  # 连接第2个点和第3个点
        [3, 7],   # 连接第3个点和第0个点
    ])

    # 创建 LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # 设置线段颜色（可选）
    line_set.paint_uniform_color([0, 1, 0])  # 绿色


    # 显示点云和坐标轴
    o3d.visualization.draw_geometries(
        [pcd, axis, line_set],
        window_name="点云查看器",
        point_show_normal=False,
        width=960,
        height=720,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()
