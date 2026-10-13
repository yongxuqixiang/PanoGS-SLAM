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

import math
import time
import torch
from odgs_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh, SH2RGB
from gaussian_splatting.utils.graphics_utils import getWorld2View2


def _compute_viewmatrix_and_campos(R, t, translate=None, scale=1.0):
    """
    优化版本：同时计算 viewmatrix 和 camera_center，避免重复的矩阵求逆。
    
    getWorld2View2 内部已经计算了相机中心，但函数只返回矩阵。
    通过这个函数，我们可以同时获取两个值，避免额外的 inverse() 调用。
    
    优化点：
    1. 复用 getWorld2View2 的逻辑，但保留中间结果（相机中心）
    2. 避免默认参数的设备转换开销（translate默认为None，在函数内创建）
    3. 所有操作都在同一设备上，避免不必要的设备转换
    4. 减少GPU同步点
    """
    device = R.device
    dtype = R.dtype
    
    # 在正确的设备上创建 translate（如果为None或需要转换）
    if translate is None:
        # 直接在目标设备上创建，避免CPU->GPU转换
        translate = torch.zeros(3, device=device, dtype=dtype)
    elif not isinstance(translate, torch.Tensor):
        translate = torch.tensor(translate, device=device, dtype=dtype)
    else:
        # 仅在必要时转换设备，使用non_blocking可能有所帮助
        if translate.device != device:
            translate = translate.to(device=device, dtype=dtype, non_blocking=True)
        elif translate.dtype != dtype:
            translate = translate.to(dtype=dtype)
    
    # 构建 W2C 矩阵（World-to-Camera）
    Rt = torch.zeros((4, 4), device=device, dtype=dtype)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    # 第一次求逆：得到 C2W（Camera-to-World）
    C2W = torch.linalg.inv(Rt)
    
    # 从 C2W 中提取相机中心（C2W 的第4列前3个元素）
    # 这是我们已经需要的结果！避免重复计算
    cam_center = C2W[:3, 3]
    # 应用 translate 和 scale 变换（与 getWorld2View2 保持一致）
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # 第二次求逆：得到最终的 W2C 矩阵（更新后的）
    Rt = torch.linalg.inv(C2W)
    
    # 返回 W2C 矩阵（需要transpose以匹配期望格式）和相机中心
    viewmatrix = Rt.transpose(0, 1)
    return viewmatrix, cam_center

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    pixel_mask=None,
    gaussian_mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 注意：tracking gaussians不参与梯度计算，所以screenspace_points只针对mapping gaussians
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # start_time = time.time()
    # 优化：直接从 R 和 T 计算 viewmatrix 和 campos，复用中间结果
    # 这样只需要2次矩阵求逆（在 _compute_viewmatrix_and_campos 中），
    # 而不是之前的：world_view_transform (2次) + camera_center 再次访问 world_view_transform (2次) + inverse() (1次) = 5次
    # 确保 R 和 T 是 float32 类型
    R_float = viewpoint_camera.R.to(dtype=torch.float32)
    T_float = viewpoint_camera.T.to(dtype=torch.float32)
    viewmatrix, campos = _compute_viewmatrix_and_campos(
        R_float, 
        T_float
    )
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        # tanfovx=tanfovx,
        # tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        # projmatrix=viewpoint_camera.full_proj_transform,
        # projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    # print(f"时间2: {time.time() - start_time:.6f}秒")
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 确保所有传递给 rasterizer 的参数都是 float32 类型
    # 先保存mapping gaussians的数量，用于后续的visibility_filter切片
    n_mapping = pc.get_xyz.shape[0]
    means3D = pc.get_xyz.to(dtype=torch.float32)
    # screenspace_points是(N, 3)形状，rasterizer内部只使用前2个维度，但需要保持(N, 3)形状以便梯度正确传播
    means2D = screenspace_points.to(dtype=torch.float32)
    opacity = pc.get_opacity.to(dtype=torch.float32)
    
    # 处理tracking gaussians（如果提供）
    tracking_gaussians = getattr(pc, 'get_tracking_gaussians_for_render', lambda: None)()
    if tracking_gaussians is not None:
        # 拼接mapping和tracking gaussians
        tracking_xyz = tracking_gaussians["xyz"].to(dtype=torch.float32)
        tracking_opacity = pc.opacity_activation(tracking_gaussians["opacity"]).to(dtype=torch.float32)
        
        means3D = torch.cat([means3D, tracking_xyz], dim=0)
        opacity = torch.cat([opacity, tracking_opacity], dim=0)
        
        # 扩展means2D以匹配新的数量（tracking gaussians不需要梯度，但需要保持(N, 3)形状）
        n_tracking = tracking_xyz.shape[0]
        tracking_means2D = torch.zeros((n_tracking, 3), dtype=torch.float32, device=means3D.device)
        means2D = torch.cat([means2D, tracking_means2D], dim=0)
    else:
        tracking_gaussians = None
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        mapping_cov3D = pc.get_covariance(scaling_modifier).to(dtype=torch.float32)
        if tracking_gaussians is not None:
            # 对于tracking gaussians，也需要计算cov3D（使用固定参数）
            tracking_scales = pc.scaling_activation(tracking_gaussians["scaling"]).to(dtype=torch.float32)
            tracking_rotations = pc.rotation_activation(tracking_gaussians["rotation"]).to(dtype=torch.float32)
            tracking_cov3D = pc.build_covariance_from_scaling_rotation(
                tracking_scales, scaling_modifier, tracking_rotations
            )
            cov3D_precomp = torch.cat([mapping_cov3D, tracking_cov3D], dim=0)
        else:
            cov3D_precomp = mapping_cov3D
    else:
        # check if the covariance is isotropic
        mapping_scales = pc.get_scaling.to(dtype=torch.float32)
        if mapping_scales.shape[-1] == 1:
            mapping_scales = mapping_scales.repeat(1, 3)
        mapping_rotations = pc.get_rotation.to(dtype=torch.float32)
        
        if tracking_gaussians is not None:
            # 拼接tracking gaussians的scales和rotations
            tracking_scales = pc.scaling_activation(tracking_gaussians["scaling"]).to(dtype=torch.float32)
            tracking_rotations = pc.rotation_activation(tracking_gaussians["rotation"]).to(dtype=torch.float32)
            
            scales = torch.cat([mapping_scales, tracking_scales], dim=0)
            rotations = torch.cat([mapping_rotations, tracking_rotations], dim=0)
        else:
            scales = mapping_scales
            rotations = mapping_rotations
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if colors_precomp is None:
        if pipe.convert_SHs_python:
            # Mapping gaussians: SH -> RGB
            mapping_features = pc.get_features.to(dtype=torch.float32)
            shs_view = mapping_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            ).to(dtype=torch.float32)
            camera_center_float = viewpoint_camera.camera_center.to(dtype=torch.float32)
            mapping_xyz = pc.get_xyz.to(dtype=torch.float32)
            dir_pp = (mapping_xyz - camera_center_float.repeat(mapping_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            mapping_colors = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            mapping_colors = torch.clamp_min(mapping_colors + 0.5, 0.0).to(dtype=torch.float32)
            
            if tracking_gaussians is not None:
                # Tracking gaussians: SH degree 0 only (DC)
                tracking_features_dc = tracking_gaussians["features_dc"].to(dtype=torch.float32)  # (N, 1, 3)
                # 对于DC only，直接使用SH2RGB转换
                tracking_colors = SH2RGB(tracking_features_dc.squeeze(1))  # (N, 3)
                tracking_colors = torch.clamp_min(tracking_colors, 0.0).to(dtype=torch.float32)
                
                colors_precomp = torch.cat([mapping_colors, tracking_colors], dim=0)
            else:
                colors_precomp = mapping_colors
        else:
            # 使用rasterizer进行SH -> RGB转换
            mapping_features = pc.get_features.to(dtype=torch.float32)  # (N, SH_coeffs, 3)
            if tracking_gaussians is not None:
                # Tracking gaussians: SH degree 0 only，需要扩展到与mapping gaussians相同的SH格式
                tracking_features_dc = tracking_gaussians["features_dc"].to(dtype=torch.float32)  # (N, 1, 3)
                n_tracking = tracking_features_dc.shape[0]
                sh_dim = (pc.max_sh_degree + 1) ** 2
                # 创建完整的SH特征（DC + zeros for rest），形状要与mapping_features一致: (N, SH_coeffs, 3)
                tracking_features_full = torch.zeros(
                    (n_tracking, sh_dim, 3), dtype=torch.float32, device=tracking_features_dc.device
                )
                tracking_features_full[:, 0, :] = tracking_features_dc.squeeze(1)  # DC component: (N, 3)
                
                shs = torch.cat([mapping_features, tracking_features_full], dim=0)
            else:
                shs = mapping_features
    else:
        colors_precomp = override_color.to(dtype=torch.float32) if override_color is not None else None

    # Prepare pixel mask for rendering if available

    if pixel_mask is None and getattr(viewpoint_camera, "mask", None) is not None:
        pixel_mask = viewpoint_camera.mask
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device=means3D.device, dtype=torch.float32).contiguous()
        if viewpoint_camera.mask is not None:
            mask_float = viewpoint_camera.mask.to(device=means3D.device, dtype=torch.float32)
            pixel_mask = pixel_mask * mask_float
        # pixel_mask = viewpoint_camera.mask.to(device=means3D.device, dtype=torch.float32).contiguous()
    # Prepare gaussian mask for rendering if available
    # gaussian_mask is a boolean tensor indicating which gaussians to render
    if gaussian_mask is not None:
        gaussian_mask = gaussian_mask.to(device=means3D.device, dtype=torch.float32).contiguous()
    # start_time = time.time()
    # myq
    # 确保 theta 和 rho 也是 float32 类型
    # 处理 cam_rot_delta 和 cam_trans_delta 可能为 None 的情况（例如在 clean() 后）
    if viewpoint_camera.cam_rot_delta is not None:
        theta = viewpoint_camera.cam_rot_delta.to(dtype=torch.float32)
    else:
        # 如果为 None，使用零向量（默认值）
        theta = torch.zeros(3, dtype=torch.float32, device=means3D.device)
    
    if viewpoint_camera.cam_trans_delta is not None:
        rho = viewpoint_camera.cam_trans_delta.to(dtype=torch.float32)
    else:
        # 如果为 None，使用零向量（默认值）
        rho = torch.zeros(3, dtype=torch.float32, device=means3D.device)
    rendered_image, depth, acc, radii, psi, lat, lon, opacity, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        theta=theta,
        rho=rho,
        pixel_mask=pixel_mask,
        gaussian_mask=gaussian_mask,  # 传递 CUDA mask
    )
    # print(f"时间1: {time.time() - start_time:.6f}秒")
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # myq
    # visibility_filter和n_touched只包含mapping gaussians的部分（排除tracking gaussians）
    visibility_filter = (radii > 0)[:n_mapping]  # 只取mapping gaussians的部分
    n_touched_mapping = n_touched[:n_mapping]  # 只取mapping gaussians的部分
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth": depth,
        # "acc": acc,
        # "psi": psi,
        # "lat": lat,
        # "lon": lon,
        "opacity": opacity,
        "n_touched": n_touched_mapping,  # 只包含mapping gaussians的部分
    }