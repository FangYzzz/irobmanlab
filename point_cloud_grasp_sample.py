import sys

from giga.perception import *
from giga.utils.implicit import get_mesh_pose_list_from_world, as_mesh#, get_scene_from_mesh_pose_list
from giga.grasp_sampler import GpgGraspSamplerPcl
from giga.utils import visual
from open3d.visualization import draw_plotly
import trimesh
from vgn.utils.transform import Rotation, Transform
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


def grasppart_rgbd_to_pc(rgb_image, depth_image, fx, fy, cx_cam, cy_cam, rect_cx, rect_cy, rect_w, rect_h):
    if rgb_image.shape[2] == 4:                                                           # astronaut(512,512,3) ZED(1080,1920,4)
        rgb_image = rgb_image[:, :, :3]

    extrinsic_rotation = R.from_quat([-0.999, 0, 0, -0.044]).as_matrix()
    extrinsic_translation = np.array([0, 0, 0.75])  
    depth_image = depth_image / 1000.0

    # Calculate the start and end points of the rectangle
    height, width = depth_image.shape
    x_start = int((rect_cx - rect_w / 2) * width)
    x_end = int((rect_cx + rect_w / 2) * width)
    y_start = int((rect_cy - rect_h / 2) * height)
    y_end = int((rect_cy + rect_h / 2) * height)
    
    # Extract the depth image and RGB image within the rectangular frame
    depth_patch = depth_image[y_start:y_end, x_start:x_end]
    rgb_patch = rgb_image[y_start:y_end, x_start:x_end, :]
    
    # Get the pixel coordinates within the rectangular frame
    u, v = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))

    # Convert pixel coordinates to robot coordinates
    Z = depth_patch
    X = (u - cx_cam) * Z / fx
    Y = (v - cy_cam) * Z / fy
    
    # Combine RGB values ​​with robot coordinates
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    points = (extrinsic_rotation @ points.T).T + extrinsic_translation
    colors = rgb_patch.reshape(-1, 3) / 255.0                                            # Colors are normalized to [0, 1]

    # Remove black points (colors close to [0, 0, 0])
    color_threshold = 0.15                                                               ### teapot 0.2
    non_black_mask = np.any(colors > color_threshold, axis=1)
    points = points[non_black_mask]
    colors = colors[non_black_mask]

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud


def draw_point_cloud(rect_cx, rect_cy, rect_w, rect_h):
    # rgb_image_path =  "/home/yuan/Mani-GPT/camera_capture/lab/ZED_mug.png"                #####
    rgb_image_path =  "/home/yuan/Mani-GPT/camera_capture/lab/ZED_image5.png"                #####
    rgb_image = o3d.io.read_image(rgb_image_path)
    rgb_image = np.asarray(rgb_image)

    # depth_image_path = "/home/yuan/Mani-GPT/camera_capture/lab/Depth_mug.png"                  #####
    depth_image_path = "/home/yuan/Mani-GPT/camera_capture/lab/Depth_5.png"                  #####
    depth_image = o3d.io.read_image(depth_image_path)
    depth_image = np.asarray(depth_image)


    fx, fy = 1057.35, 1056.91
    cx_cam, cy_cam = 1082.06, 637.621
    rect_cx = rect_cx                                    # 裁剪区域中心点x坐标
    rect_cy = rect_cy * 1920.0 / 1080.0                  # 裁剪区域中心点y坐标
    rect_w = rect_w                                      # 裁剪区域宽度
    rect_h = rect_h * 1920.0 / 1080.0                    # 裁剪区域高度
    # rect_cx, rect_cy, rect_w, rect_h = 0.7165774, 0.20537513 * 1920.0 / 1080.0, 0.054115403, 0.055510864 * 1920.0 / 1080.0

    point_cloud = grasppart_rgbd_to_pc(rgb_image, depth_image, fx, fy, cx_cam, cy_cam, rect_cx, rect_cy, rect_w, rect_h)

    draw_plotly([point_cloud])                           # display point cloud on web page
    o3d.visualization.draw_geometries([point_cloud])

    return point_cloud


def get_GraspSample(rect_cx, rect_cy, rect_w, rect_h):
    point_cloud = draw_point_cloud(rect_cx, rect_cy, rect_w, rect_h)

    num_parallel_workers = 1
    num_grasps = 10

    sampler = GpgGraspSamplerPcl(0.05)              # Franka finger depth is actually a little less than 0.05m

    grasps, grasps_pos, grasps_rot = sampler.sample_grasps_parallel(point_cloud, 
                                                                    num_parallel=num_parallel_workers,
                                                                    num_grasps=num_grasps, 
                                                                    max_num_samples=80,
                                                                    safety_dis_above_table=0.01,
                                                                    show_final_grasps=False)

    print("len of grasps", len(grasps))

    grasps_scene = trimesh.Scene()

    grasp_mesh_list = [visual.grasp2mesh(g, score=1) for g in grasps]
    for i, g_mesh in enumerate(grasp_mesh_list):
        print(i)
        grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')

    draw_plotly([point_cloud, as_mesh(grasps_scene).as_open3d])

    print("grasps position, grasps rotation:", grasps_pos[0], grasps_rot[0])

    return grasps_pos[0], grasps_rot[0]




# rect_cx = 0.7556685611605644
# rect_cy = 0.13180365469306707
# rect_w = 0.13613322414457799
# rect_h = 0.033825541567057374

# get_GraspSample(rect_cx, rect_cy, rect_w, rect_h)






# # Already have depth image and camera intrinsics
# # cropped_rgb_image_path =  "/home/yuan/Mani-GPT/camera_capture/lab/ZED_image5.png"                   # 00000
# # cropped_rgb_image_path =  "/home/yuan/Mani-GPT/camera_capture/lab/cropped_rgb_mug.png"              #####
# cropped_rgb_image_path =  "/home/yuan/Mani-GPT/camera_capture/lab/cropped_rgb_5_pepper.png"

# # cropped_depth_image_path = "/home/yuan/Mani-GPT/camera_capture/lab/Depth_5.png"                     # 00000
# # cropped_depth_image_path = "/home/yuan/Mani-GPT/camera_capture/lab/cropped_depth_mug.png"           #####
# cropped_depth_image_path = "/home/yuan/Mani-GPT/camera_capture/lab/cropped_depth_5_pepper.png"

# rgb_image = o3d.io.read_image(cropped_rgb_image_path)
# rgb_image = np.asarray(rgb_image)

# depth_image = o3d.io.read_image(cropped_depth_image_path)
# depth_image = np.asarray(depth_image)



# extrinsic = Transform(Rotation.from_quat([-0.999, 0, 0, -0.044]), [0, 1, 0.71])        # transformation the camera frame to the robot frame
# # Transform(Rotation.from_quat([-1, 0, 0, 0]), [0, 0, 0.43])  Transform(Rotation.from_quat([-0.996, 0, 0, -0.087]), [0, 0, 0.45])
# intrinsic_matrix = np.array([[1057.35, 0, 1082.06],
#                              [0, 1056.91, 637.621],
#                              [0, 0, 1]])                                               # Intrinsic camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]



# def create_tsdf_from_depth(depth_image,rgb_image, intrinsic_matrix, voxel_length, sdf_trunc):
    
#     # 创建TSDF体素网格
#     tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
#         voxel_length=4.0 / 512.0,
#         sdf_trunc=sdf_trunc,
#         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
#         volume_unit_resolution=100,  # 体素网格的初始分辨率
#         depth_sampling_stride=  1   # 深度图像采样步长
#     )
#     if rgb_image.shape[2] == 4:        # astronaut(512,512,3) ZED(1080,1920,4)
#         rgb_image = rgb_image[:, :, :3]
#     # 创建RGBD图像
#     rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#         o3d.geometry.Image(rgb_image.astype(np.uint8)),  # 假设没有RGB数据
#         o3d.geometry.Image(depth_image.astype(np.uint8)),
#         depth_scale=1.0,  # 根据实际情况设置
#         depth_trunc=1000.0,  # 根据实际情况设置
#         convert_rgb_to_intensity=False
#     )
#     # rgbd_image = rgb_image

#     height, width = depth_image.shape
#     intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 
#                                                   intrinsic_matrix[0, 0], 
#                                                   intrinsic_matrix[1, 1], 
#                                                   intrinsic_matrix[0, 2], 
#                                                   intrinsic_matrix[1, 2])

    
#     extrinsic = Transform(Rotation.from_quat([-0.940, 0, 0, -0.342]), [0, 0, 0.43])
#     extrinsic = extrinsic.as_matrix()

#     tsdf_volume.integrate(rgbd_image, intrinsic, extrinsic)                 
    
#     return tsdf_volume

# def extract_mesh_from_tsdf(tsdf_volume):
#     mesh = tsdf_volume.extract_triangle_mesh()
#     # mesh = tsdf_volume.extract_point_cloud()
#     return mesh

# def visualize_mesh(mesh):
#     o3d.visualization.draw_geometries([mesh])


# # # 创建TSDF体素网格
# # voxel_length = 0.01                  # 体素的边长
# # sdf_trunc = 1.0                      # 截断距离
# # tsdf_volume = create_tsdf_from_depth(depth_image, rgb_image,intrinsic_matrix, voxel_length, sdf_trunc)

# # # 提取并可视化网格
# # mesh = extract_mesh_from_tsdf(tsdf_volume)
# # visualize_mesh(mesh)

# # # near = 0.05
# # # far = 5.0


# rgb_image = o3d.geometry.Image((rgb_image).astype(np.uint8))
# depth_img_np = o3d.geometry.Image(depth_image)
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_img_np, convert_rgb_to_intensity=False)  # A RGB-D image is a combination of a RGB image and its corresponding depth image

# intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1082, 1057.35, 1056.91, 1082.06, 637.621)
# extrinsic = extrinsic.as_matrix()                                                                                         # 通常为 4x4 的单位矩阵
# point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic)
# # point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_img_np, intrinsic, extrinsic)

# # o3d.visualization.draw_geometries([point_cloud])


# # remove the desktop point
# low = point_cloud.get_min_bound()
# high = point_cloud.get_max_bound()
# # low[2] += 0.03
# print(low)
# print(high)
# bounding_box = o3d.geometry.AxisAlignedBoundingBox(low, high)
# pc = point_cloud.crop(bounding_box)
# # print(point_cloud)
# # draw_plotly([point_cloud])                    # use o3d

# o3d.visualization.draw_geometries([pc])
# draw_plotly([pc])                               # display point cloud on web page

# # denoised_cloud = pc.voxel_down_sample(voxel_size=0.05)
# # denoised_cloud = o3d.geometry.PointCloud.remove_statistical_outlier(denoised_cloud,nb_neighbors=20,std_ratio=2.0)

# # smoothed_cloud = pc.smooth_laplacian(alpha=0.99)
# # o3d.visualization.draw_geometries([denoised_cloud])


# plane_model, inliers = pc.segment_plane(distance_threshold=0.01,
#                                         ransac_n=3,
#                                         num_iterations=4000)
# # 模型参数
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
# # 平面内的点
# inlier_cloud = pc.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# # 平面外的点
# outlier_cloud = pc.select_by_index(inliers, invert=True)


# # low = outlier_cloud.get_min_bound()
# # high = outlier_cloud.get_max_bound()
# # low[2]=0.05
# # print(low)
# # print(high)
# # bounding_box = o3d.geometry.AxisAlignedBoundingBox(low, high)
# # pc = outlier_cloud.crop(bounding_box)

# o3d.visualization.draw_geometries([outlier_cloud])

# # def display_inlier_outlier(cloud, ind):
# #     inlier_cloud = cloud.select_by_index(ind)
# #     outlier_cloud = cloud.select_by_index(ind, invert=True)

# #     # print("Showing outliers (red) and inliers (gray): ")
# #     # outlier_cloud.paint_uniform_color([1, 0, 0])
# #     # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
# #     o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
# # print("Radius oulier removal")

# # for i in range(1000):
# #     outlier_cloud, ind = outlier_cloud.remove_radius_outlier(nb_points=16, radius=0.05)


# # # display_inlier_outlier(outlier_cloud, ind)
# # o3d.visualization.draw_geometries([outlier_cloud])




# num_parallel_workers = 1
# num_grasps = 10
# safety_dist_above_table = 0.05

# sampler = GpgGraspSamplerPcl(0.05)              # Franka finger depth is actually a little less than 0.05m

# grasps, grasps_pos, grasps_rot = sampler.sample_grasps_parallel(pc, 
#                                                                 num_parallel=num_parallel_workers,
#                                                                 num_grasps=num_grasps, 
#                                                                 max_num_samples=80,
#                                                                 safety_dis_above_table=0.01,
#                                                                 show_final_grasps=False)
# # print("grasps_pos:", grasps_pos[0])
# # print("grasps_rot:", grasps_rot[0])
# print("len of grasps", len(grasps))

# grasps_scene = trimesh.Scene()

# grasp_mesh_list = [visual.grasp2mesh(g, score=1) for g in grasps]
# for i, g_mesh in enumerate(grasp_mesh_list):
#     print(i)
#     grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')

# draw_plotly([pc, as_mesh(grasps_scene).as_open3d])


# # ply_file_path = "/home/yuan/Mani-GPT/camera_capture/lab/Cloud_mug.ply"
# # pcd = o3d.io.read_point_cloud(ply_file_path)
# # # draw_plotly([pcd]) 
# # # print(pcd)
# # o3d.visualization.draw_geometries([pcd], window_name="PLY Point Cloud Visualization")



