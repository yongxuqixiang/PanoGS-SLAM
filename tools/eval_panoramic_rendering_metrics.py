#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panoramic rendering-quality evaluation.
Computes PSNR / SSIM / LPIPS for equirectangular panoramas.
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import json

# Evaluation metrics
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
from utils.camera_utils import Camera
from argparse import Namespace
from plyfile import PlyData
import torch.nn.functional as F


# ============ Panoramic camera parameters ============
PANORAMIC_CAMERA_PARAMS = {
    "width": 1024,         # panorama width
    "height": 512,         # panorama height
    "fovx": 2 * np.pi,     # horizontal FoV: 360 deg
    "fovy": np.pi,         # vertical FoV: 180 deg
}

# Equivalent focal lengths (for the panoramic projection)
PANORAMIC_CAMERA_PARAMS["fx"] = PANORAMIC_CAMERA_PARAMS["width"] / PANORAMIC_CAMERA_PARAMS["fovx"]
PANORAMIC_CAMERA_PARAMS["fy"] = PANORAMIC_CAMERA_PARAMS["height"] / PANORAMIC_CAMERA_PARAMS["fovy"]
PANORAMIC_CAMERA_PARAMS["cx"] = PANORAMIC_CAMERA_PARAMS["width"] / 2.0
PANORAMIC_CAMERA_PARAMS["cy"] = PANORAMIC_CAMERA_PARAMS["height"] / 2.0
# =========================================


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix.
    quat: [qx, qy, qz, qw]
    """
    qx, qy, qz, qw = quat
    
    # Normalize the quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Build the rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    return R


def load_camera_trajectory(traj_file: str) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Load a TUM-format camera trajectory.
    Format: timestamp tx ty tz qx qy qz qw
    Returns: {timestamp: (R_w2c, t_w2c)}
    """
    poses = {}
    with open(traj_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            # TUM stores C2W; convert to W2C
            R_c2w = quaternion_to_rotation_matrix(np.array([qx, qy, qz, qw]))
            t_c2w = np.array([tx, ty, tz])
            
            # Build the C2W matrix
            T_c2w = np.eye(4)
            T_c2w[:3, :3] = R_c2w
            T_c2w[:3, 3] = t_c2w
            
            # Invert to get the W2C matrix
            T_w2c = np.linalg.inv(T_c2w)
            
            # Extract the W2C rotation and translation
            R_w2c = T_w2c[:3, :3]
            t_w2c = T_w2c[:3, 3]
            
            poses[timestamp] = (R_w2c, t_w2c)
    
    return poses


def load_images_from_directory(image_dir: str) -> Dict[int, str]:
    """
    Load images from a directory, sorted by filename.
    
    Args:
        image_dir: image directory
    
    Returns:
        {index: image_path} dict
    """
    image_map = {}
    
    # Supported image formats
    valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    # Collect the image files
    image_files = []
    for f in os.listdir(image_dir):
        if f.endswith(valid_extensions):
            image_files.append(f)
    
    # Sort by filename
    image_files.sort()
    
    # Build the mapping
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        image_map[idx] = image_path
    
    return image_map


def detect_sh_degree(ply_file: str) -> int:
    """
    Auto-detect the spherical-harmonics degree in a PLY file.
    """
    plydata = PlyData.read(ply_file)
    
    # Collect all f_rest_* attributes
    extra_f_names = [
        p.name
        for p in plydata.elements[0].properties
        if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
    
    num_f_rest = len(extra_f_names)
    
    if num_f_rest == 0:
        sh_degree = 0
    else:
        sh_degree = int(np.sqrt((num_f_rest + 3) / 3) - 1)
    
    print(f"Detected {num_f_rest} f_rest fields in the PLY, inferred sh_degree: {sh_degree}")
    return sh_degree


def load_gaussian_model(ply_file: str, device: str = "cuda") -> GaussianModel:
    """
    Load a 3D Gaussian Splatting model.
    """
    print(f"Loading 3DGS model: {ply_file}")
    
    # Auto-detect sh_degree
    sh_degree = detect_sh_degree(ply_file)
    
    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(ply_file)
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    return gaussians


def create_panoramic_camera(
    R: np.ndarray, 
    t: np.ndarray, 
    uid: int,
    width: int,
    height: int,
    fovx: float,
    fovy: float,
    device: str = "cuda"
) -> Camera:
    """
    Create a panoramic camera.
    R, t: W2C (world-to-camera) rotation and translation
    """
    # Compute the equivalent focal lengths
    fx = width / fovx
    fy = height / fovy
    cx = width / 2.0
    cy = height / 2.0
    
    # Build the projection matrix
    projection_matrix = getProjectionMatrix2(
        znear=0.01, 
        zfar=100.0, 
        fx=fx, 
        fy=fy, 
        cx=cx, 
        cy=cy, 
        W=width, 
        H=height
    ).transpose(0, 1).to(device)
    
    # Build the pose matrix (4x4)
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    gt_pose = torch.from_numpy(pose).float().to(device)
    
    # Create the camera object
    camera = Camera(
        uid=uid,
        color=None,
        depth=None,
        gt_T=gt_pose,
        projection_matrix=projection_matrix,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fovx=fovx,
        fovy=fovy,
        image_height=height,
        image_width=width,
        device=device
    )
    
    # Set the camera pose
    camera.update_RT(
        torch.from_numpy(R).float().to(device),
        torch.from_numpy(t).float().to(device)
    )
    
    return camera


def render_panoramic_image(
    camera: Camera, 
    gaussians: GaussianModel, 
    pipe_args: Namespace, 
    bg_color: torch.Tensor
) -> torch.Tensor:
    """
    Render a panoramic image.
    """
    with torch.no_grad():
        render_pkg = render(
            camera, 
            gaussians, 
            pipe_args, 
            bg_color,
            scaling_modifier=1.0,
            override_color=None
        )
        
        if render_pkg is None:
            return None
            
        image = render_pkg["render"]
        image = torch.clamp(image, 0.0, 1.0)
    
    return image


def load_gt_panoramic_image(
    image_path: str, 
    width: int, 
    height: int, 
    device: str = "cuda"
) -> torch.Tensor:
    """
    Load a ground-truth panorama.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"failed to load image: {image_path}")
    
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height))
    
    # To tensor [C, H, W]
    img = torch.from_numpy(img).float().to(device) / 255.0
    img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    return img


def load_mask(
    mask_path: str,
    width: int,
    height: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Load a mask image.
    White (255) marks excluded regions; black (0) marks kept regions.
    
    Returns:
        mask: [H, W] bool tensor, True=keep, False=exclude
    """
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"failed to load mask: {mask_path}")
    
    # Resize
    if mask_img.shape[0] != height or mask_img.shape[1] != width:
        mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # To bool tensor: white (255) -> False (excluded), black (0) -> True (kept)
    mask = torch.from_numpy(mask_img).float().to(device)
    mask = mask < 127.5  # < 127.5 counts as black (kept), >= 127.5 as white (excluded)
    
    return mask


def equirectangular_to_pinhole(
    equirect_img: torch.Tensor,
    yaw: float,
    pitch: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int
) -> torch.Tensor:
    """
    Project an equirectangular panorama onto a pinhole viewport.
    
    Args:
        equirect_img: [C, H_pano, W_pano] panorama
        yaw: horizontal rotation (rad), 0 = forward
        pitch: pitch angle (rad), 0 = level
        fx, fy: pinhole focal lengths
        cx, cy: pinhole principal point
        width, height: pinhole image size
    
    Returns:
        pinhole_img: [C, height, width] pinhole projection
    """
    device = equirect_img.device
    C, H_pano, W_pano = equirect_img.shape
    
    # Pixel grid of the pinhole image
    u = torch.arange(0, width, dtype=torch.float32, device=device)
    v = torch.arange(0, height, dtype=torch.float32, device=device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing='xy')
    
    # Pinhole camera: pixel coords -> normalized camera coords
    x_cam = (u_grid - cx) / fx
    y_cam = (v_grid - cy) / fy
    z_cam = torch.ones_like(x_cam)
    
    # Normalize to unit vectors
    ray_length = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_norm = x_cam / ray_length
    y_norm = y_cam / ray_length
    z_norm = z_cam / ray_length
    
    # Apply the rotation (yaw and pitch)
    # yaw: rotation about Y (left/right)
    # pitch: rotation about X (up/down)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    
    # Apply pitch first (about X)
    x_rot1 = x_norm
    y_rot1 = y_norm * cos_pitch - z_norm * sin_pitch
    z_rot1 = y_norm * sin_pitch + z_norm * cos_pitch
    
    # Then yaw (about Y)
    x_world = x_rot1 * cos_yaw + z_rot1 * sin_yaw
    y_world = y_rot1
    z_world = -x_rot1 * sin_yaw + z_rot1 * cos_yaw
    
    # Convert to spherical coordinates
    # theta: horizontal angle [-pi, pi]
    # phi: vertical angle [-pi/2, pi/2]
    theta = torch.atan2(x_world, z_world)  # horizontal angle
    phi = torch.asin(torch.clamp(y_world, -1.0, 1.0))  # vertical angle
    
    # Map to panorama uv coordinates
    # theta: [-pi, pi] -> [0, W_pano]
    # phi: [-pi/2, pi/2] -> [0, H_pano]
    u_pano = (theta + np.pi) / (2 * np.pi) * W_pano
    v_pano = (np.pi / 2 - phi) / np.pi * H_pano
    
    # Normalize to [-1, 1] for grid_sample
    u_pano_norm = (u_pano / W_pano) * 2 - 1
    v_pano_norm = (v_pano / H_pano) * 2 - 1
    
    # grid_sample expects [N, H, W, 2]
    grid = torch.stack([u_pano_norm, v_pano_norm], dim=-1).unsqueeze(0)
    
    # Sample
    equirect_batch = equirect_img.unsqueeze(0)  # [1, C, H_pano, W_pano]
    pinhole_img = F.grid_sample(
        equirect_batch, 
        grid, 
        mode='bilinear', 
        padding_mode='border',
        align_corners=False
    )
    
    pinhole_img = pinhole_img.squeeze(0)  # [C, height, width]
    
    return pinhole_img


def sample_multiple_viewports(
    equirect_img: torch.Tensor,
    pinhole_params: Dict,
    num_viewports: int = 3
) -> List[torch.Tensor]:
    """
    Sample multiple pinhole viewports from a panorama.
    
    Args:
        equirect_img: [C, H, W] panorama
        pinhole_params: pinhole camera parameter dict
        num_viewports: number of viewports
    
    Returns:
        viewports: list of [C, H, W] pinhole images
    """
    viewports = []
    
    # Evenly distribute the yaw angles
    yaw_angles = [2 * np.pi * i / num_viewports for i in range(num_viewports)]
    pitch = 0.0  # level
    
    for yaw in yaw_angles:
        viewport = equirectangular_to_pinhole(
            equirect_img,
            yaw=yaw,
            pitch=pitch,
            fx=pinhole_params['fx'],
            fy=pinhole_params['fy'],
            cx=pinhole_params['cx'],
            cy=pinhole_params['cy'],
            width=pinhole_params['width'],
            height=pinhole_params['height']
        )
        viewports.append(viewport)
    
    return viewports


def compute_metrics(
    pred: torch.Tensor, 
    gt: torch.Tensor, 
    lpips_model, 
    use_height_mask: bool = False,
    use_viewport_psnr: bool = False,
    pinhole_params: Dict = None,
    num_viewports: int = 3,
    mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    pred, gt: [C, H, W], values in [0, 1]
    use_height_mask: mask out the top/bottom quarters (non-viewport mode only)
    use_viewport_psnr: compute PSNR on multi-view pinhole projections
    pinhole_params: pinhole camera params (required when use_viewport_psnr=True)
    num_viewports: number of sampled viewports
    mask: [H, W] bool tensor, True=keep, False=exclude (overrides use_height_mask)
    """
    # Add the batch dimension
    pred_batch = pred.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    
    if use_viewport_psnr:
        # Option C: viewport-based PSNR
        # Project the panorama onto several pinhole viewports and average the PSNR
        if pinhole_params is None:
            raise ValueError("pinhole_params is required when use_viewport_psnr=True")
        
        # Sample viewports for pred and gt
        pred_viewports = sample_multiple_viewports(pred, pinhole_params, num_viewports)
        gt_viewports = sample_multiple_viewports(gt, pinhole_params, num_viewports)
        
        # PSNR per viewport
        psnr_values = []
        for pred_vp, gt_vp in zip(pred_viewports, gt_viewports):
            pred_vp_batch = pred_vp.unsqueeze(0)
            gt_vp_batch = gt_vp.unsqueeze(0)
            psnr_vp = psnr(pred_vp_batch, gt_vp_batch, data_range=1.0).item()
            psnr_values.append(psnr_vp)
        
        # Average PSNR
        psnr_val = np.mean(psnr_values)
        
        # SSIM and LPIPS are still computed on the full panorama
        ssim_val = ssim(pred_batch, gt_batch, data_range=1.0).item()
        lpips_val = lpips_model(pred_batch, gt_batch).item()
        
        # Return the extra viewport PSNR info
        metrics = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "viewport_psnr_values": psnr_values,
            "num_viewports": num_viewports
        }
        
    elif mask is not None:
        # Use the custom mask (highest priority)
        C, H, W = pred.shape
        
        # Check that the mask has valid regions
        valid_pixels = mask.sum().item()
        if valid_pixels == 0:
            raise ValueError("mask has no valid region (everything is excluded)")
        
        # Apply the mask, keeping only True regions
        # Expand the mask to match the channel dimension
        mask_expanded = mask.unsqueeze(0).expand(C, -1, -1)  # [C, H, W]
        
        # Extract valid pixels
        pred_masked = pred[mask_expanded].reshape(C, -1)
        gt_masked = gt[mask_expanded].reshape(C, -1)
        
        # SSIM/LPIPS need a rectangular image
        # Find the mask bounding box
        mask_indices = torch.where(mask)
        if len(mask_indices[0]) == 0:
            raise ValueError("mask has no valid region")
        
        h_min, h_max = mask_indices[0].min().item(), mask_indices[0].max().item() + 1
        w_min, w_max = mask_indices[1].min().item(), mask_indices[1].max().item() + 1
        
        # Crop to the bounding box
        pred_crop = pred[:, h_min:h_max, w_min:w_max]
        gt_crop = gt[:, h_min:h_max, w_min:w_max]
        mask_crop = mask[h_min:h_max, w_min:w_max]
        
        # Zero out excluded regions (does not affect SSIM/LPIPS)
        mask_crop_expanded = mask_crop.unsqueeze(0).expand(C, -1, -1)
        pred_crop = pred_crop * mask_crop_expanded.float()
        gt_crop = gt_crop * mask_crop_expanded.float()
        
        # PSNR (valid pixels only)
        mse = torch.mean((pred_masked - gt_masked) ** 2)
        if mse == 0:
            psnr_val = float('inf')
        else:
            psnr_val = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)).item()
        
        # SSIM (cropped image)
        pred_crop_batch = pred_crop.unsqueeze(0)
        gt_crop_batch = gt_crop.unsqueeze(0)
        ssim_val = ssim(pred_crop_batch, gt_crop_batch, data_range=1.0).item()
        
        # LPIPS (cropped image)
        lpips_val = lpips_model(pred_crop_batch, gt_crop_batch).item()
        
        metrics = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "valid_pixels": valid_pixels
        }
    elif use_height_mask:
        # Height mask: exclude the top/bottom quarters
        C, H, W = pred.shape
        height_mask = torch.zeros((H, W), dtype=torch.bool, device=pred.device)
        # Keep the middle half (1/4 to 3/4)
        h_start = H // 4
        h_end = 3 * H // 4
        height_mask[h_start:h_end, :] = True
        
        # Apply the mask
        pred_masked = pred[:, height_mask].reshape(C, -1)
        gt_masked = gt[:, height_mask].reshape(C, -1)
        
        # Reshape to an image for SSIM and LPIPS
        # PSNR can use the masked region directly
        mask_height = h_end - h_start
        pred_masked_img = pred[:, h_start:h_end, :]
        gt_masked_img = gt[:, h_start:h_end, :]
        
        # PSNR (masked region)
        pred_batch_masked = pred_masked_img.unsqueeze(0)
        gt_batch_masked = gt_masked_img.unsqueeze(0)
        psnr_val = psnr(pred_batch_masked, gt_batch_masked, data_range=1.0).item()
        
        # SSIM (masked region)
        ssim_val = ssim(pred_batch_masked, gt_batch_masked, data_range=1.0).item()
        
        # LPIPS (masked region)
        lpips_val = lpips_model(pred_batch_masked, gt_batch_masked).item()
        
        metrics = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val
        }
    else:
        # Compute PSNR
        psnr_val = psnr(pred_batch, gt_batch, data_range=1.0).item()
        
        # Compute SSIM
        ssim_val = ssim(pred_batch, gt_batch, data_range=1.0).item()
        
        # Compute LPIPS
        lpips_val = lpips_model(pred_batch, gt_batch).item()
        
        metrics = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val
        }
    
    return metrics


def save_comparison_image(
    pred_np: np.ndarray,
    gt_np: np.ndarray,
    metrics: Dict[str, float],
    save_path: str,
    idx: int,
    use_height_mask: bool = False,
    use_viewport_psnr: bool = False,
    mask_np: np.ndarray = None
):
    """
    Save a comparison figure (rendered + GT stacked vertically).
    use_height_mask: annotate the masked regions
    use_viewport_psnr: whether viewport PSNR is used
    mask_np: [H, W] mask array, True=keep, False=exclude
    """
    h, w = pred_np.shape[:2]
    
    # Stack vertically
    comparison = np.zeros((h * 2 + 10, w, 3), dtype=np.uint8)
    comparison[:h, :] = pred_np
    comparison[h:h+10, :] = 255  # white separator
    comparison[h+10:, :] = gt_np
    
    # With a custom mask, overlay the excluded regions
    if mask_np is not None:
        # mask_np: True=keep, False=exclude
        overlay = comparison.copy()
        
        # Red overlay for excluded regions
        mask_vis = (~mask_np).astype(np.uint8) * 255  # False (excluded) -> 255
        
        # Mark on the rendered image
        for y in range(h):
            for x in range(w):
                if not mask_np[y, x]:
                    overlay[y, x] = [0, 0, 255]  # red mark
        
        # Mark on the GT image
        for y in range(h):
            for x in range(w):
                if not mask_np[y, x]:
                    overlay[h + 10 + y, x] = [0, 0, 255]  # red mark
        
        # Blend
        alpha = 0.4
        comparison = cv2.addWeighted(comparison, 1 - alpha, overlay, alpha, 0)
    
    # With a height mask, overlay the excluded regions
    elif use_height_mask and not use_viewport_psnr:
        h_start = h // 4
        h_end = 3 * h // 4
        
        # Semi-transparent red overlay
        overlay = comparison.copy()
        # Top excluded region
        cv2.rectangle(overlay, (0, 0), (w, h_start), (0, 0, 255), -1)
        # Bottom excluded region
        cv2.rectangle(overlay, (0, h_end), (w, h), (0, 0, 255), -1)
        # Same for the GT image
        cv2.rectangle(overlay, (0, h + 10), (w, h + 10 + h_start), (0, 0, 255), -1)
        cv2.rectangle(overlay, (0, h + 10 + h_end), (w, h * 2 + 10), (0, 0, 255), -1)
        
        # Blend
        alpha = 0.3
        comparison = cv2.addWeighted(comparison, 1 - alpha, overlay, alpha, 0)
        
        # Add boundary lines
        cv2.line(comparison, (0, h_start), (w, h_start), (0, 255, 255), 2)
        cv2.line(comparison, (0, h_end), (w, h_end), (0, 255, 255), 2)
        cv2.line(comparison, (0, h + 10 + h_start), (w, h + 10 + h_start), (0, 255, 255), 2)
        cv2.line(comparison, (0, h + 10 + h_end), (w, h + 10 + h_end), (0, 255, 255), 2)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Rendered", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Ground Truth", (10, h + 40), font, 1, (0, 255, 0), 2)
    
    # Add metric info
    if use_viewport_psnr:
        psnr_note = f" (avg of {metrics.get('num_viewports', 3)} viewports)"
    elif mask_np is not None:
        valid_ratio = mask_np.sum() / mask_np.size
        psnr_note = f" (custom mask, {valid_ratio*100:.1f}% valid)"
    elif use_height_mask:
        psnr_note = " (masked region)"
    else:
        psnr_note = ""
    
    metrics_text = f"PSNR: {metrics['psnr']:.2f} dB{psnr_note} | SSIM: {metrics['ssim']:.4f} | LPIPS: {metrics['lpips']:.4f}"
    cv2.putText(comparison, metrics_text, (10, h * 2 - 10), font, 0.6, (0, 255, 255), 2)
    
    cv2.imwrite(save_path, comparison)


def main():
    parser = argparse.ArgumentParser(description="Panoramic rendering-quality evaluation")
    parser.add_argument("--traj_file", type=str, required=True,
                        help="camera trajectory file (TUM format)")
    parser.add_argument("--ply_file", type=str, required=True,
                        help="3DGS point cloud (PLY) file")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="panorama image directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output directory")
    parser.add_argument("--interval", type=int, default=5,
                        help="evaluate every N frames")
    parser.add_argument("--width", type=int, default=1024,
                        help="panorama width")
    parser.add_argument("--height", type=int, default=512,
                        help="panorama height")
    parser.add_argument("--device", type=str, default="cuda",
                        help="device")
    parser.add_argument("--save_images", action="store_true",
                        help="save rendered images")
    parser.add_argument("--save_gt", action="store_true",
                        help="save GT images")
    parser.add_argument("--save_comparison", action="store_true",
                        help="save comparison images")
    parser.add_argument("--bg_color", type=str, default="white",
                        choices=["white", "black"],
                        help="background color")
    parser.add_argument("--use_height_mask", action="store_true",
                        help="exclude the top/bottom quarters (poles)")
    parser.add_argument("--use_viewport_psnr", action="store_true",
                        help="viewport-based PSNR on pinhole projections")
    parser.add_argument("--num_viewports", type=int, default=3,
                        help="number of viewports (for viewport PSNR)")
    parser.add_argument("--viewport_fov_h", type=float, default=120.0,
                        help="viewport horizontal FOV (deg)")
    parser.add_argument("--viewport_fov_v", type=float, default=90.0,
                        help="viewport vertical FOV (deg)")
    parser.add_argument("--mask_file", type=str, default=None,
                        help="mask file (white=exclude, black=evaluate)")
    
    args = parser.parse_args()
    
    # Use the configured panoramic camera parameters
    camera_params = PANORAMIC_CAMERA_PARAMS.copy()
    if args.width is not None:
        camera_params["width"] = args.width
        camera_params["fx"] = camera_params["width"] / camera_params["fovx"]
        camera_params["cx"] = camera_params["width"] / 2.0
    if args.height is not None:
        camera_params["height"] = args.height
        camera_params["fy"] = camera_params["height"] / camera_params["fovy"]
        camera_params["cy"] = camera_params["height"] / 2.0
    
    # Pinhole camera parameters (viewport-based PSNR)
    pinhole_params = None
    if args.use_viewport_psnr:
        pinhole_params = {
            "fx": 277.12812921102045,
            "fy": 415.6921938165307,
            "cx": 480.0,
            "cy": 240.0,
            "width": 960,
            "height": 480
        }
        print(f"\nPinhole viewport parameters:")
        print(f"  resolution: {pinhole_params['width']} x {pinhole_params['height']}")
        print(f"  focal: fx={pinhole_params['fx']:.2f}, fy={pinhole_params['fy']:.2f}")
        print(f"  principal point: cx={pinhole_params['cx']:.2f}, cy={pinhole_params['cy']:.2f}")
        print(f"  viewports: {args.num_viewports}")
        print(f"  viewport FOV: {args.viewport_fov_h} x {args.viewport_fov_v} deg")
    
    # Load the mask if provided
    mask_tensor = None
    if args.mask_file is not None:
        if not os.path.exists(args.mask_file):
            print(f"error: mask file not found: {args.mask_file}")
            sys.exit(1)
        print(f"\nLoading mask: {args.mask_file}")
        try:
            mask_tensor = load_mask(
                args.mask_file,
                camera_params['width'],
                camera_params['height'],
                device=args.device
            )
            valid_ratio = mask_tensor.sum().item() / (mask_tensor.shape[0] * mask_tensor.shape[1])
            print(f"OK: mask loaded, valid region: {valid_ratio*100:.2f}%")
        except Exception as e:
            print(f"error: failed to load mask: {e}")
            sys.exit(1)
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories for saved images
    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "rendered"), exist_ok=True)
    if args.save_gt:
        os.makedirs(os.path.join(args.output_dir, "ground_truth"), exist_ok=True)
    if args.save_comparison:
        os.makedirs(os.path.join(args.output_dir, "comparison"), exist_ok=True)
    
    print("=" * 60)
    print("Panoramic rendering quality evaluation")
    print("=" * 60)
    print("Evaluation config:")
    print(f"  trajectory: {args.traj_file}")
    print(f"  point cloud: {args.ply_file}")
    print(f"  image dir: {args.image_dir}")
    print(f"  mask: {args.mask_file if args.mask_file else 'none (full image)'}")
    print(f"  interval: every {args.interval} frames")
    print(f"  background: {args.bg_color}")
    print(f"  PSNR method: {'viewport-based pinhole projection' if args.use_viewport_psnr else 'direct panorama'}")
    if not args.use_viewport_psnr and not args.mask_file:
        print(f"  height mask: {'yes (top/bottom quarters)' if args.use_height_mask else 'no'}")
    print(f"  save rendered: {'yes' if args.save_images else 'no'}")
    print(f"  save GT: {'yes' if args.save_gt else 'no'}")
    print(f"  save comparison: {'yes' if args.save_comparison else 'no'}")
    print(f"\nPanoramic camera parameters:")
    print(f"  resolution: {camera_params['width']} x {camera_params['height']}")
    if args.use_height_mask:
        h_start = camera_params['height'] // 4
        h_end = 3 * camera_params['height'] // 4
        print(f"  evaluated region: rows {h_start}-{h_end} (middle half)")
    print(f"  FOV: {np.degrees(camera_params['fovx']):.1f}° x {np.degrees(camera_params['fovy']):.1f}°")
    print(f"  equivalent focal: fx={camera_params['fx']:.2f}, fy={camera_params['fy']:.2f}")
    print(f"  principal point: cx={camera_params['cx']:.2f}, cy={camera_params['cy']:.2f}")
    print("=" * 60)
    
    # Load the camera trajectory
    print("\n[1/4] Loading camera trajectory...")
    poses = load_camera_trajectory(args.traj_file)
    print(f"OK: loaded {len(poses)} poses")
    
    # Load the image list
    print("\n[2/4] Loading images...")
    image_map = load_images_from_directory(args.image_dir)
    print(f"OK: loaded {len(image_map)} images")
    
    # Load the 3DGS model
    print("\n[3/4] Loading the 3DGS model...")
    gaussians = load_gaussian_model(args.ply_file, device=args.device)
    print(f"OK: model loaded")
    
    # Create the LPIPS model
    print("\n[4/4] Initializing metrics...")
    lpips_model = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", 
        normalize=True
    ).to(args.device)
    print(f"OK: LPIPS model initialized")
    
    # Create the render pipeline parameters
    pipe_args = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False
    )
    
    # Background color
    if args.bg_color == "white":
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=args.device)
    else:
        bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=args.device)
    
    # Evaluate
    print("\nEvaluating...")
    print("=" * 60)
    
    timestamps = sorted(poses.keys())
    # Match the number of images and poses
    num_frames = min(len(poses), len(image_map))
    eval_indices = list(range(0, num_frames, args.interval))
    
    results = []
    failed_frames = []
    
    for idx in tqdm(eval_indices, desc="evaluating"):
        # Get the pose
        if idx >= len(timestamps):
            print(f"\nwarning: index {idx} out of trajectory range, skipping")
            continue
        
        timestamp = timestamps[idx]
        R, t = poses[timestamp]
        
        # Get the GT image path
        if idx not in image_map:
            print(f"\nwarning: no GT image for index {idx}, skipping")
            failed_frames.append(idx)
            continue
        
        gt_path = image_map[idx]
        if not os.path.exists(gt_path):
            print(f"\nwarning: GT image not found: {gt_path}, skipping")
            failed_frames.append(idx)
            continue
        
        # Load the GT image
        try:
            gt_image = load_gt_panoramic_image(
                gt_path, 
                camera_params['width'], 
                camera_params['height'],
                device=args.device
            )
        except Exception as e:
            print(f"\nerror: failed to load GT image ({gt_path}): {e}, skipping")
            failed_frames.append(idx)
            continue
        
        # Create the panoramic camera
        try:
            camera = create_panoramic_camera(
                R, t, idx,
                camera_params['width'],
                camera_params['height'],
                camera_params['fovx'],
                camera_params['fovy'],
                device=args.device
            )
        except Exception as e:
            print(f"\nerror: failed to create the camera: {e}, skipping")
            failed_frames.append(idx)
            continue
        
        # Render the panorama
        try:
            pred_image = render_panoramic_image(camera, gaussians, pipe_args, bg_color)
        except Exception as e:
            print(f"\nerror: rendering failed: {e}, skipping")
            failed_frames.append(idx)
            continue
        
        if pred_image is None:
            print(f"\nwarning: rendering returned None at index {idx}, skipping")
            failed_frames.append(idx)
            continue
        
        # With a mask, set excluded regions to the background color
        if mask_tensor is not None:
            mask_expanded = mask_tensor.unsqueeze(0).expand_as(pred_image)
            # Set excluded regions (mask=False) to the background color
            pred_image = torch.where(mask_expanded, pred_image, bg_color.view(3, 1, 1))
        
        # Compute metrics
        try:
            metrics = compute_metrics(
                pred_image, 
                gt_image, 
                lpips_model, 
                use_height_mask=args.use_height_mask if (not args.use_viewport_psnr and mask_tensor is None) else False,
                use_viewport_psnr=args.use_viewport_psnr,
                pinhole_params=pinhole_params,
                num_viewports=args.num_viewports,
                mask=mask_tensor
            )
        except Exception as e:
            print(f"\nerror: metric computation failed: {e}, skipping")
            failed_frames.append(idx)
            continue
        
        result = {
            "frame_idx": idx,
            "timestamp": float(timestamp),
            "gt_image_path": gt_path,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"]
        }
        
        # With viewport PSNR, store the per-viewport values
        if args.use_viewport_psnr and "viewport_psnr_values" in metrics:
            result["viewport_psnr_values"] = metrics["viewport_psnr_values"]
            result["num_viewports"] = metrics["num_viewports"]
        
        results.append(result)
        
        # Save images
        if args.save_images or args.save_gt or args.save_comparison:
            # Rendered image to numpy
            pred_np = (pred_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_np = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
            
            # GT image to numpy
            gt_np = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_np = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
            
            # Save the rendered image
            if args.save_images:
                save_path = os.path.join(args.output_dir, "rendered", f"render_{idx:04d}.png")
                cv2.imwrite(save_path, pred_np)
            
            # Save the GT image
            if args.save_gt:
                save_path = os.path.join(args.output_dir, "ground_truth", f"gt_{idx:04d}.png")
                cv2.imwrite(save_path, gt_np)
            
            # Save the comparison image
            if args.save_comparison:
                save_path = os.path.join(args.output_dir, "comparison", f"comp_{idx:04d}.png")
                mask_np = mask_tensor.cpu().numpy() if mask_tensor is not None else None
                save_comparison_image(
                    pred_np, gt_np, metrics, save_path, idx, 
                    use_height_mask=args.use_height_mask if mask_tensor is None else False,
                    use_viewport_psnr=args.use_viewport_psnr,
                    mask_np=mask_np
                )
    
    print("\n" + "=" * 60)
    print("Evaluation done!")
    print("=" * 60)
    
    # Average metrics
    if len(results) > 0:
        avg_psnr = np.mean([r["psnr"] for r in results])
        avg_ssim = np.mean([r["ssim"] for r in results])
        avg_lpips = np.mean([r["lpips"] for r in results])
        
        std_psnr = np.std([r["psnr"] for r in results])
        std_ssim = np.std([r["ssim"] for r in results])
        std_lpips = np.std([r["lpips"] for r in results])
        
        print("\nResults:")
        print(f"  evaluated frames: {len(results)}")
        print(f"  failed frames: {len(failed_frames)}")
        print(f"\n  PSNR:  {avg_psnr:.4f} +/- {std_psnr:.4f} dB")
        print(f"  SSIM:  {avg_ssim:.4f} +/- {std_ssim:.4f}")
        print(f"  LPIPS: {avg_lpips:.4f} +/- {std_lpips:.4f}")
        
        # Save the results
        summary = {
            "camera_type": "panoramic",
            "camera_params": camera_params,
            "mask_file": args.mask_file,
            "use_custom_mask": args.mask_file is not None,
            "psnr_method": "viewport_based" if args.use_viewport_psnr else "direct",
            "use_viewport_psnr": args.use_viewport_psnr,
            "use_height_mask": args.use_height_mask if (not args.use_viewport_psnr and args.mask_file is None) else False,
            "height_mask_range": f"{camera_params['height']//4}-{3*camera_params['height']//4}" if (args.use_height_mask and not args.use_viewport_psnr and args.mask_file is None) else None,
            "num_frames_evaluated": len(results),
            "num_frames_failed": len(failed_frames),
            "failed_frame_indices": failed_frames,
            "metrics": {
                "mean_psnr": float(avg_psnr),
                "std_psnr": float(std_psnr),
                "mean_ssim": float(avg_ssim),
                "std_ssim": float(std_ssim),
                "mean_lpips": float(avg_lpips),
                "std_lpips": float(std_lpips)
            },
            "detailed_results": results
        }
        
        # With viewport PSNR, add the related config
        if args.use_viewport_psnr:
            summary["pinhole_params"] = pinhole_params
            summary["num_viewports"] = args.num_viewports
            summary["viewport_fov"] = {
                "horizontal": args.viewport_fov_h,
                "vertical": args.viewport_fov_v
            }
        
        output_json = os.path.join(args.output_dir, "evaluation_results.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"\nOK: results saved to: {output_json}")
        
        # Print saved-image info
        if args.save_images or args.save_gt or args.save_comparison:
            print("\nSaved images:")
            if args.save_images:
                print(f"  rendered: {os.path.join(args.output_dir, 'rendered')} ({len(results)})")
            if args.save_gt:
                print(f"  GT: {os.path.join(args.output_dir, 'ground_truth')} ({len(results)})")
            if args.save_comparison:
                print(f"  comparison: {os.path.join(args.output_dir, 'comparison')} ({len(results)})")
    else:
        print("\nERROR: no frame was successfully evaluated!")
        print(f"failed frames: {len(failed_frames)}")
        if failed_frames:
            print(f"failed indices: {failed_frames[:10]}{'...' if len(failed_frames) > 10 else ''}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
