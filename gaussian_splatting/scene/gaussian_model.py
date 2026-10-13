#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import cv2
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from scipy.special import erf
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        self.isotropic = False

        # Tracking Gaussians buffer (MVP)
        # 存储临时的tracking-only gaussians，不参与mapping/densify/prune
        self.tracking_gaussians_xyz = torch.empty(0, device="cuda")
        self.tracking_gaussians_features_dc = torch.empty(0, device="cuda")  # SH degree 0 only
        self.tracking_gaussians_scaling = torch.empty(0, device="cuda")
        self.tracking_gaussians_rotation = torch.empty(0, device="cuda")
        self.tracking_gaussians_opacity = torch.empty(0, device="cuda")
        self.tracking_gaussians_life = torch.empty(0, device="cuda", dtype=torch.int32)  # 剩余生命周期（帧数）

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # myq
    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None, last_render_opacity=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            # myq
            # print("depthmap is not None")
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
            depth_np = np.asarray(depth)
            # print("Depth shape:", depth_np.shape)
            # print("Depth min/max:", np.nanmin(depth_np), np.nanmax(depth_np))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                print("depth_raw is None")
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                print("randn depth generation")
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))
        
        # return self.create_panoramic_pointcloud(cam, rgb, depth, init)
        return self.create_panoramic_pointcloud_with_opacity(cam, rgb, depth, init, last_render_opacity)
        # return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    # myq
    # 全景相机高斯点生成
    def create_panoramic_pointcloud(self, cam, rgb, depth, init = False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]

        point_size = self.config["Dataset"]["point_size"]

        # 将Open3D图像转换为NumPy数组
        depth_np = np.asarray(depth)
        rgb_np = np.asarray(rgb)
        
        # 自适应pointsize
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))

        # 生成球面坐标网格
        theta = np.linspace(-np.pi, np.pi, cam.image_width)
        phi = np.linspace(np.pi/2, -np.pi/2, cam.image_height) 
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        r = depth_np.flatten()
        valid_mask = r > 0
        
        # phi_flat = phi_grid.flatten()
        # area_weight = np.cos(phi_flat)          # [0, 1]
        # area_weight = np.clip(area_weight, 0, None)

        # rand = np.random.rand(len(area_weight))
        # keep_mask = rand < (area_weight / area_weight.max())

        # valid_mask = valid_mask & keep_mask
        
        if cam.mask is not None:
            mask_np = cam.mask.cpu().numpy()
            mask_np = mask_np.astype(bool)
            valid_mask = valid_mask & mask_np.flatten()


        r_valid = r[valid_mask]
        
        # 球面坐标转笛卡尔坐标（右手系：x右，y上，z前）
        # phi: 从+90度(顶部)到-90度(底部)
        # theta: 从-180度到+180度
        x = r_valid * np.cos(phi_grid.flatten()[valid_mask]) * np.sin(theta_grid.flatten()[valid_mask])
        y = -r_valid * np.sin(phi_grid.flatten()[valid_mask])
        z = r_valid * np.cos(phi_grid.flatten()[valid_mask]) * np.cos(theta_grid.flatten()[valid_mask])
        
        # 获取有效颜色
        colors = (rgb_np.reshape(-1, 3)[valid_mask] / 255.0).astype(np.float32)

        # 创建点云
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(np.vstack((x, y, z)).T)
        pcd_tmp.colors = o3d.utility.Vector3dVector(colors)


        # 点云坐标旋转
        Rt = torch.eye(4, device=cam.R.device)
        Rt[:3, :3] = cam.R
        Rt[:3, 3] = cam.T
        C2W = torch.linalg.inv(Rt).cpu().numpy()
        pcd_tmp.transform(C2W)
        
        # 保存点云可视化（用于调试深度估计质量）
        if hasattr(self, 'config') and self.config.get("depth_estimation", {}).get("save_visualization", False):
            try:
                import os
                from PIL import Image
                import cv2
                
                viz_dir = os.path.join(self.config.get("Results", {}).get("save_dir", "results"), "depth_visualization", "pointclouds")
                os.makedirs(viz_dir, exist_ok=True)
                
                # 保存原始点云（相机坐标系）
                pcd_cam = o3d.geometry.PointCloud()
                pcd_cam.points = o3d.utility.Vector3dVector(np.vstack((x, y, z)).T)
                pcd_cam.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(os.path.join(viz_dir, f"pointcloud_cam_{cam.uid}.ply"), pcd_cam)
                
                # 保存世界坐标系点云
                o3d.io.write_point_cloud(os.path.join(viz_dir, f"pointcloud_world_{cam.uid}.ply"), pcd_tmp)
                
                # 保存深度图可视化
                depth_vis = depth_np.copy()
                depth_vis[~valid_mask.reshape(depth_np.shape)] = 0
                depth_normalized = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
                depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)).save(
                    os.path.join(viz_dir, f"depth_visualization_{cam.uid}.png")
                )
                
                # 保存RGB图像
                rgb_vis = (rgb_np * 255).astype(np.uint8)
                Image.fromarray(rgb_vis).save(os.path.join(viz_dir, f"rgb_{cam.uid}.png"))
                
                print(f"Point cloud visualization saved for camera {cam.uid}")
                
            except Exception as e:
                print(f"Failed to save point cloud visualization: {e}")


        # 下采样
        if downsample_factor > 1:
            pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)

        # 获取高斯新点云
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        pcd = BasicPointCloud(
            points=new_xyz, 
            colors=new_rgb, 
            normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd
        fused_point_cloud = torch.tensor(new_xyz, dtype=torch.float32, device="cuda")
        fused_color = RGB2SH(torch.tensor(new_rgb, dtype=torch.float32, device="cuda"))
        # 初始化高斯参数
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            dtype=torch.float32, device="cuda"
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # 计算尺度和不透明度
        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        # scales = 1.0
        scale = torch.clamp(scale, min = 1.0)
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        opacities = inverse_sigmoid(
            0.5 * torch.ones(
                (fused_point_cloud.shape[0], 1), 
                dtype=torch.float32, 
                device="cuda"
            )
        )

        # 可视化第一帧的点云（用于论文插图）
        if init:
            self._visualize_first_frame_pointcloud(pcd_tmp, cam)

        return fused_point_cloud, features, scales, rots, opacities

    def _visualize_first_frame_pointcloud(self, pcd, cam):
        """
        保存第一帧的点云为PCD格式，供pcl_viewer查看
        """
        try:
            import os
            
            # 创建保存目录
            viz_dir = os.path.join(
                self.config.get("Results", {}).get("save_dir", "results"),
                "first_frame_visualization"
            )
            os.makedirs(viz_dir, exist_ok=True)
            
            # 保存为PCD格式（PCL标准格式，Open3D会根据扩展名自动识别格式）
            pcd_file = os.path.join(viz_dir, "first_frame_pointcloud.pcd")
            o3d.io.write_point_cloud(pcd_file, pcd)
            print(f"第一帧点云已保存到: {pcd_file}")
            print(f"点云包含 {len(pcd.points)} 个点")
            print(f"可以使用 pcl_viewer {pcd_file} 查看点云")
            
        except Exception as e:
            print(f"保存第一帧点云时出错: {e}")
            import traceback
            traceback.print_exc()

    def create_panoramic_pointcloud_with_opacity(
        self, cam, rgb, depth, init=False, last_render_opacity=None
    ):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]

        point_size = self.config["Dataset"]["point_size"]

        depth_np = np.asarray(depth)
        rgb_np = np.asarray(rgb)

        
        under_modeled_mask = None
        opacity_boost = self.config["Dataset"].get("opacity_region_boost", 2.0)

        opacity_np = None
        if last_render_opacity is not None:
            try:
                if torch.is_tensor(last_render_opacity):
                    opacity_np = last_render_opacity.detach().cpu().numpy()
                else:
                    opacity_np = np.asarray(last_render_opacity)
                if opacity_np.shape[1:3] == (cam.image_height, cam.image_width):
                    opacity_threshold = self.config["Dataset"].get(
                        "opacity_threshold", 0.3
                    )

                    under_modeled_mask = (
                        (opacity_np < opacity_threshold) &
                        (depth_np > 0)
                    ).flatten()
                    print("number of under-modeled points: ", under_modeled_mask.sum())

            except Exception as e:
                print(f"[WARN] Failed to process opacity map: {e}")
                under_modeled_mask = None
                opacity_np = None

        if self.config["Dataset"].get("adaptive_pointsize", False):
            point_size = min(0.05, point_size * np.median(depth_np[depth_np > 0]))

        theta = np.linspace(-np.pi, np.pi, cam.image_width)
        phi = np.linspace(np.pi / 2, -np.pi / 2, cam.image_height)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        r = depth_np.flatten()
        valid_mask = r > 0
        phi_flat = phi_grid.flatten()
        area_weight = np.cos(phi_flat)          # [0, 1]
        area_weight = np.clip(area_weight, 0, None)

        rand = np.random.rand(len(area_weight))
        keep_mask = rand < (area_weight / area_weight.max())

        valid_mask = valid_mask & keep_mask
        if cam.mask is not None:
            valid_mask &= cam.mask.cpu().numpy().astype(bool).flatten()

        r_valid = r[valid_mask]

        x = r_valid * np.cos(phi_grid.flatten()[valid_mask]) * np.sin(theta_grid.flatten()[valid_mask])
        y = -r_valid * np.sin(phi_grid.flatten()[valid_mask])
        z = r_valid * np.cos(phi_grid.flatten()[valid_mask]) * np.cos(theta_grid.flatten()[valid_mask])

        colors = (rgb_np.reshape(-1, 3)[valid_mask] / 255.0).astype(np.float32)

        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(np.vstack((x, y, z)).T)
        pcd_tmp.colors = o3d.utility.Vector3dVector(colors)

        Rt = torch.eye(4, device=cam.R.device)
        Rt[:3, :3] = cam.R
        Rt[:3, 3] = cam.T
        C2W = torch.linalg.inv(Rt).cpu().numpy()
        pcd_tmp.transform(C2W)


        if downsample_factor > 1:
            if under_modeled_mask is not None:
                valid_under_mask = under_modeled_mask[valid_mask]

                pts = np.asarray(pcd_tmp.points)
                cols = np.asarray(pcd_tmp.colors)

                under_pts = pts[valid_under_mask]
                under_cols = cols[valid_under_mask]
                well_pts = pts[~valid_under_mask]
                well_cols = cols[~valid_under_mask]

                under_ds = max(1.0, downsample_factor / opacity_boost)
                well_ds = downsample_factor

                def downsample(pts, cols, factor):
                    if len(pts) == 0 or factor <= 1:
                        return pts, cols
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.colors = o3d.utility.Vector3dVector(cols)
                    pcd = pcd.random_down_sample(1.0 / factor)
                    return np.asarray(pcd.points), np.asarray(pcd.colors)

                under_pts, under_cols = downsample(under_pts, under_cols, under_ds)
                well_pts, well_cols = downsample(well_pts, well_cols, well_ds)

                pts = np.vstack([under_pts, well_pts]) if len(well_pts) else under_pts
                cols = np.vstack([under_cols, well_cols]) if len(well_cols) else under_cols

                pcd_tmp.points = o3d.utility.Vector3dVector(pts)
                pcd_tmp.colors = o3d.utility.Vector3dVector(cols)
            else:
                pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)

        # frame_id = getattr(cam, "uid", getattr(cam, "frame_id", "unknown"))
        # save_root = self.config.get("Results", {}).get("save_dir", "results")

        # # 保存当前帧 opacity 热力图（如果提供了 last_render_opacity）
        # if opacity_np is not None:
        #     try:
        #         opacity_map = np.asarray(opacity_np).squeeze()
        #         if opacity_map.shape == (cam.image_height, cam.image_width):
        #             opacity_vis_dir = os.path.join(save_root, "opacity_visualization")
        #             os.makedirs(opacity_vis_dir, exist_ok=True)
        #             opacity_norm = np.clip(opacity_map, 0.0, 1.0)
        #             opacity_heatmap = cv2.applyColorMap(
        #                 (opacity_norm * 255).astype(np.uint8),
        #                 cv2.COLORMAP_JET,
        #             )
        #             heatmap_path = os.path.join(opacity_vis_dir, f"Frame_{frame_id}_opacity.png")
        #             cv2.imwrite(heatmap_path, opacity_heatmap)
        #         else:
        #             print(
        #                 f"[WARN] Skip opacity heatmap: unexpected shape {opacity_map.shape}, "
        #                 f"expected {(cam.image_height, cam.image_width)}"
        #             )
        #     except Exception as e:
        #         print(f"[WARN] Failed to save opacity heatmap for frame {frame_id}: {e}")

        # # 保存当前帧新插入点云
        # try:
        #     pcd_save_dir = os.path.join(save_root, "inserted_pointclouds")
        #     os.makedirs(pcd_save_dir, exist_ok=True)
        #     pcd_path = os.path.join(pcd_save_dir, f"Frame_{frame_id}.pcd")
        #     o3d.io.write_point_cloud(pcd_path, pcd_tmp)
        # except Exception as e:
            # print(f"[WARN] Failed to save inserted pointcloud for frame {frame_id}: {e}")

        # ===============================
        # 6️⃣ Gaussian initialization（原逻辑）
        # ===============================
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        fused_point_cloud = torch.tensor(new_xyz, dtype=torch.float32, device="cuda")
        fused_color = RGB2SH(torch.tensor(new_rgb, dtype=torch.float32, device="cuda"))

        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
            device="cuda"
        )
        features[:, :3, 0] = fused_color

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(new_xyz).float().cuda()),
            1e-7
        ) * point_size

        scales = torch.log(torch.sqrt(dist2))[..., None]
        # scales = torch.clamp(scales, min = 1.0)
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.8 * torch.ones((fused_point_cloud.shape[0], 1), device="cuda")
        )
        # print("number of points: ", fused_point_cloud.shape[0])
        if init:
            self._visualize_first_frame_pointcloud(pcd_tmp, cam)
        return fused_point_cloud, features, scales, rots, opacities



    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, last_render_opacity=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap, last_render_opacity=last_render_opacity)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    # myq
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        xyz = self.get_xyz  # (N, 3)
        norm = torch.norm(xyz, dim=-1).clamp(min=1e-5)
        z = xyz[:, 2]
        # pos_weight = torch.abs(z / norm)
        # lat = torch.acos(z / norm)
        # center = torch.pi / 2
        # sigma = torch.pi / 6  
        # pos_weight = torch.exp(-((lat - center) ** 2) / (2 * sigma ** 2)) 
        # adaptive_threshold = grad_threshold * (1.2 - pos_weight)
        # selected_pts_mask = padded_grad >= adaptive_threshold
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        xyz = self.get_xyz  # (N, 3)
        norm = torch.norm(xyz, dim=-1).clamp(min=1e-5)
        z = xyz[:, 2]
        # 余弦权重
        # pos_weight = torch.abs(z / norm)

        # lat = torch.acos(z / norm)
        # center = torch.pi / 2
        # sigma = torch.pi / 6  
        # pos_weight = torch.exp(-((lat - center) ** 2) / (2 * sigma ** 2)) 
        # adaptive_threshold = grad_threshold * (1.2 - pos_weight)
        # selected_pts_mask = padded_grad >= adaptive_threshold

        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    # ==================== Tracking Gaussians (MVP) ====================
    def inject_tracking_gaussians(
        self, 
        viewpoint_camera, 
        opacity_map, 
        depth_map, 
        rgb_map, 
        max_samples=500,
        opacity_threshold=0.15,
        life_time_frames=5
    ):
        """
        在低opacity区域注入临时的tracking gaussians
        
        Args:
            viewpoint_camera: 相机视图对象
            opacity_map: 渲染的opacity map (1, H, W) 或 (H, W)
            depth_map: 深度图 (1, H, W) 或 (H, W)
            rgb_map: RGB图像 (3, H, W) 或 (H, W, 3)
            max_samples: 最多采样的像素数 (200-500)
            opacity_threshold: opacity阈值 (默认0.15)
            life_time_frames: 生命周期（帧数，默认5）
        """
        if opacity_map is None or depth_map is None or rgb_map is None:
            return
        
        # 转换为numpy并处理维度
        if torch.is_tensor(opacity_map):
            opacity_np = opacity_map.detach().cpu().numpy()
        else:
            opacity_np = np.asarray(opacity_map)
        
        if torch.is_tensor(depth_map):
            depth_np = depth_map.detach().cpu().numpy()
        else:
            depth_np = np.asarray(depth_map)
        
        # 处理RGB图像：应用exposure校正（与后端保持一致）
        # 后端代码：image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        if torch.is_tensor(rgb_map):
            # 应用exposure校正
            if hasattr(viewpoint_camera, 'exposure_a') and hasattr(viewpoint_camera, 'exposure_b'):
                exposure_a = torch.exp(viewpoint_camera.exposure_a)
                exposure_b = viewpoint_camera.exposure_b
                rgb_corrected = exposure_a * rgb_map + exposure_b
                rgb_corrected = torch.clamp(rgb_corrected, 0.0, 1.0)
                rgb_np = rgb_corrected.detach().cpu().numpy()
            else:
                rgb_np = rgb_map.detach().cpu().numpy()
            
            if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:  # (3, H, W)
                rgb_np = rgb_np.transpose(1, 2, 0)  # (H, W, 3)
        else:
            rgb_np = np.asarray(rgb_map)
    
        
        # 处理维度
        if opacity_np.ndim == 3:
            opacity_np = opacity_np[0] if opacity_np.shape[0] == 1 else opacity_np.squeeze()
        if depth_np.ndim == 3:
            depth_np = depth_np[0] if depth_np.shape[0] == 1 else depth_np.squeeze()
        
        H, W = opacity_np.shape
        
        # 检测低opacity区域
        low_opacity_mask = (opacity_np < opacity_threshold) & (depth_np > 0)
        
        # 应用相机mask（如果存在）
        if hasattr(viewpoint_camera, 'mask') and viewpoint_camera.mask is not None:
            cam_mask = viewpoint_camera.mask.detach().cpu().numpy()
            if cam_mask.ndim == 3:
                cam_mask = cam_mask[0] if cam_mask.shape[0] == 1 else cam_mask.squeeze()
            low_opacity_mask = low_opacity_mask & (cam_mask > 0)
        
        # 稀疏采样
        candidate_indices = np.where(low_opacity_mask.flatten())[0]
        if len(candidate_indices) == 0:
            print("out")
            return
        
        n_samples = min(len(candidate_indices), max_samples)
        sampled_indices = np.random.choice(candidate_indices, size=n_samples, replace=False)
        
        # 转换为2D坐标
        y_coords = sampled_indices // W
        x_coords = sampled_indices % W
        
        # 获取深度和RGB
        depths = depth_np[y_coords, x_coords]
        colors = rgb_np[y_coords, x_coords]  # (N, 3)
        
        
        # 确保RGB值在[0, 1]范围（应用exposure校正后应该已经是[0, 1]了）
        # 但如果输入是[0, 255]范围，则归一化
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
        

        
        # 全景相机：球面坐标转换
        theta = (x_coords / W - 0.5) * 2 * np.pi  # [-pi, pi]
        phi = (0.5 - y_coords / H) * np.pi  # [pi/2, -pi/2]
        
        x_cam = depths * np.cos(phi) * np.sin(theta)
        y_cam = -depths * np.sin(phi)
        z_cam = depths * np.cos(phi) * np.cos(theta)
        
        xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)
        
        # 转换到世界坐标系
        Rt = torch.eye(4, device=viewpoint_camera.R.device)
        Rt[:3, :3] = viewpoint_camera.R
        Rt[:3, 3] = viewpoint_camera.T
        C2W = torch.linalg.inv(Rt).cpu().numpy()
        
        xyz_cam_homo = np.hstack([xyz_cam, np.ones((xyz_cam.shape[0], 1))])  # (N, 4)
        xyz_world = (C2W @ xyz_cam_homo.T).T[:, :3]  # (N, 3)
        
        # 创建tracking gaussians参数
        n_new = xyz_world.shape[0]
        point_size = self.config["Dataset"]["point_size"] * 3
        
        # 自适应pointsize（与create_panoramic_pointcloud一致）
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depths[depths > 0]))
        
        # xyz: 世界坐标
        new_xyz = torch.tensor(xyz_world, dtype=torch.float32, device="cuda")
        
        # features_dc: RGB -> SH (DC only, SH degree 0)
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device="cuda")  # (N, 3)
        features_dc = RGB2SH(colors_tensor)  # (N, 3)
        features_dc = features_dc.unsqueeze(1)  # (N, 1, 3) for DC only
        
        # 计算尺度和不透明度（与create_panoramic_pointcloud一致）
        # 使用distCUDA2计算点之间的距离，然后计算尺度
        dist2 = (
            torch.clamp_min(
                distCUDA2(new_xyz),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)
        
        new_scaling = scales
        
        # rotation: identity quaternion [1, 0, 0, 0]
        new_rotation = torch.zeros((n_new, 4), dtype=torch.float32, device="cuda")
        new_rotation[:, 0] = 1.0
        
        # opacity: inverse_sigmoid(0.7)
        new_opacity = inverse_sigmoid(torch.tensor(0.4, device="cuda")).repeat(n_new, 1)
        
        # life_time: 初始化为life_time_frames
        new_life = torch.full((n_new,), life_time_frames, dtype=torch.int32, device="cuda")
        
        # 拼接到现有的tracking gaussians
        if self.tracking_gaussians_xyz.shape[0] == 0:
            self.tracking_gaussians_xyz = new_xyz
            self.tracking_gaussians_features_dc = features_dc
            self.tracking_gaussians_scaling = new_scaling
            self.tracking_gaussians_rotation = new_rotation
            self.tracking_gaussians_opacity = new_opacity
            self.tracking_gaussians_life = new_life
        else:
            self.tracking_gaussians_xyz = torch.cat([self.tracking_gaussians_xyz, new_xyz], dim=0)
            self.tracking_gaussians_features_dc = torch.cat([self.tracking_gaussians_features_dc, features_dc], dim=0)
            self.tracking_gaussians_scaling = torch.cat([self.tracking_gaussians_scaling, new_scaling], dim=0)
            self.tracking_gaussians_rotation = torch.cat([self.tracking_gaussians_rotation, new_rotation], dim=0)
            self.tracking_gaussians_opacity = torch.cat([self.tracking_gaussians_opacity, new_opacity], dim=0)
            self.tracking_gaussians_life = torch.cat([self.tracking_gaussians_life, new_life], dim=0)
        print(new_xyz.shape[0])
    
    def update_tracking_gaussians_lifecycle(self):
        """
        更新tracking gaussians的生命周期，删除过期的gaussians
        每帧调用一次，life -= 1，life <= 0的gaussians被删除
        """
        if self.tracking_gaussians_life.shape[0] == 0:
            return
        
        # life -= 1
        self.tracking_gaussians_life -= 1
        
        # 删除life <= 0的gaussians
        valid_mask = self.tracking_gaussians_life > 0
        
        if not valid_mask.all():
            self.tracking_gaussians_xyz = self.tracking_gaussians_xyz[valid_mask]
            self.tracking_gaussians_features_dc = self.tracking_gaussians_features_dc[valid_mask]
            self.tracking_gaussians_scaling = self.tracking_gaussians_scaling[valid_mask]
            self.tracking_gaussians_rotation = self.tracking_gaussians_rotation[valid_mask]
            self.tracking_gaussians_opacity = self.tracking_gaussians_opacity[valid_mask]
            self.tracking_gaussians_life = self.tracking_gaussians_life[valid_mask]
    
    def get_tracking_gaussians_for_render(self):
        """
        获取用于渲染的tracking gaussians数据
        返回字典，包含所有需要的参数（requires_grad=False）
        """
        if self.tracking_gaussians_xyz.shape[0] == 0:
            return None
        
        return {
            "xyz": self.tracking_gaussians_xyz.detach(),  # 不参与梯度计算
            "features_dc": self.tracking_gaussians_features_dc.detach(),
            "scaling": self.tracking_gaussians_scaling.detach(),
            "rotation": self.tracking_gaussians_rotation.detach(),
            "opacity": self.tracking_gaussians_opacity.detach(),
        }
    
    def clear_tracking_gaussians(self):
        """清空所有tracking gaussians"""
        self.tracking_gaussians_xyz = torch.empty(0, device="cuda")
        self.tracking_gaussians_features_dc = torch.empty(0, device="cuda")
        self.tracking_gaussians_scaling = torch.empty(0, device="cuda")
        self.tracking_gaussians_rotation = torch.empty(0, device="cuda")
        self.tracking_gaussians_opacity = torch.empty(0, device="cuda")
        self.tracking_gaussians_life = torch.empty(0, device="cuda", dtype=torch.int32)
