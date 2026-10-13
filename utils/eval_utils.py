import matplotlib
matplotlib.use('Agg')  # must come before other matplotlib imports
import os
import json

import cv2
import evo
import numpy as np
import torch
import torch.nn.functional as F
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log

def create_detailed_comparison_table(traj_ref, traj_est_aligned, errors, plot_dir, label):
    """
    Detailed per-coordinate comparison in pure Python.
    """
    gt_positions = traj_ref.positions_xyz
    est_positions = traj_est_aligned.positions_xyz
    
    # Create the data
    comparison_data = []
    
    for i in range(len(gt_positions)):
        gt_x, gt_y, gt_z = gt_positions[i]
        est_x, est_y, est_z = est_positions[i]
        
        error_x = est_x - gt_x
        error_y = est_y - gt_y  
        error_z = est_z - gt_z
        error_3d = errors[i]
        
        comparison_data.append([
            i, gt_x, gt_y, gt_z, est_x, est_y, est_z, 
            error_x, error_y, error_z, error_3d
        ])
    
    # Save as CSV
    csv_path = os.path.join(plot_dir, f"detailed_comparison_{label}.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("frame_id,gt_x,gt_y,gt_z,est_x,est_y,est_z,error_x,error_y,error_z,error_3d\n")
        # Data rows
        for row in comparison_data:
            f.write(",".join([f"{x:.6f}" if isinstance(x, float) else str(x) for x in row]) + "\n")
    
    # Compute statistics
    errors_x = [row[7] for row in comparison_data]
    errors_y = [row[8] for row in comparison_data] 
    errors_z = [row[9] for row in comparison_data]
    errors_3d = [row[10] for row in comparison_data]
    
    stats_summary = {
        'rmse_ate': np.mean(errors_3d),
        'mean_error_x': np.mean(errors_x),
        'mean_error_y': np.mean(errors_y),
        'mean_error_z': np.mean(errors_z),
        'std_error_x': np.std(errors_x),
        'std_error_y': np.std(errors_y),
        'std_error_z': np.std(errors_z),
        'max_error_3d': np.max(errors_3d),
        'min_error_3d': np.min(errors_3d)
    }
    
    
    return {
        'csv_path': csv_path,
        'stats_summary': stats_summary
    }

def detect_and_fix_axis_reflection(traj_est, traj_ref, correct_scale=False):
    """
    Detect and correct axis reflection issues.

    Depending on the dataset convention, the estimated trajectory may be
    mirrored w.r.t. the ground-truth frame. We evaluate the aligned ATE for
    both the identity and the y-axis-flipped variant and keep the variant
    with the smaller error (data-driven, no hardcoded convention).

    The alignment mode (rigid SE(3) vs scale-corrected Sim(3)) must match
    the one used by the final evaluation, otherwise the decision can be
    dominated by scale drift instead of the reflection itself.

    Args:
        traj_est: estimated trajectory (PosePath3D)
        traj_ref: reference trajectory (PosePath3D)
        correct_scale: use Umeyama Sim(3) alignment (True) or rigid SE(3)
            alignment (False); pass the same value used for the final ATE.

    Returns:
        corrected_traj: corrected trajectory
        reflection_info: reflection info dict
    """
    est_pos = traj_est.positions_xyz.copy()

    reflection_info = {
        'x_reflected': False,
        'y_reflected': False,
        'z_reflected': False,
        'x_correlation': 0.0,
        'y_correlation': 0.0,
        'z_correlation': 0.0,
        'reflection_applied': False,
        'rmse_keep': 0.0,
        'rmse_flip': 0.0
    }

    def _aligned_rmse(flip_y):
        pos = est_pos.copy()
        if flip_y:
            pos[:, 1] *= -1
        poses = []
        for i, pose in enumerate(traj_est.poses_se3):
            p = pose.copy()
            p[:3, 3] = pos[i]
            poses.append(p)
        cand = PosePath3D(poses_se3=poses)
        cand_aligned = trajectory.align_trajectory(
            cand, traj_ref, correct_scale=correct_scale
        )
        data = (traj_ref, cand_aligned)
        m = metrics.APE(metrics.PoseRelation.translation_part)
        m.process_data(data)
        return m.get_statistic(metrics.StatisticsType.rmse)

    rmse_keep = _aligned_rmse(False)
    rmse_flip = _aligned_rmse(True)
    reflection_info['rmse_keep'] = float(rmse_keep)
    reflection_info['rmse_flip'] = float(rmse_flip)

    print("\n========== Axis reflection check ==========")
    print(f"alignment mode: {'Sim(3)' if correct_scale else 'SE(3)'}")
    print(f"ATE RMSE (keep):    {rmse_keep:.6f} m")
    print(f"ATE RMSE (y-flip):  {rmse_flip:.6f} m")

    if rmse_flip < rmse_keep:
        est_pos[:, 1] *= -1
        reflection_info['y_reflected'] = True
        reflection_info['reflection_applied'] = True
        print("[auto] applying the y-axis reflection")
    else:
        print("[auto] keeping the original trajectory (no reflection)")
    print("=" * 40)

    # Rebuild the trajectory (with the selected variant)
    corrected_poses = []
    for i, pose in enumerate(traj_est.poses_se3):
        pose_new = pose.copy()
        pose_new[:3, 3] = est_pos[i]
        corrected_poses.append(pose_new)

    corrected_traj = PosePath3D(poses_se3=corrected_poses)
    print("axis reflection check done!\n")
    return corrected_traj, reflection_info

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, final=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    print("Estimated traj length:", len(traj_est.positions_xyz))
    print("Reference traj length:", len(traj_ref.positions_xyz))

    # Detect and fix reflections before alignment; the flip decision must
    # use the same alignment mode as the final ATE below.
    traj_est_corrected, reflection_info = detect_and_fix_axis_reflection(
        traj_est, traj_ref, correct_scale=monocular
    )
    
    traj_est_aligned = trajectory.align_trajectory(
        traj_est_corrected, traj_ref, correct_scale=monocular
    )   
    
    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    
    # Add reflection info to the statistics
    ape_stats['reflection_info'] = reflection_info
    
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")
    if reflection_info['reflection_applied']:
        reflected_axes = [axis for axis in ['x', 'y', 'z'] if reflection_info[f'{axis}_reflected']]
        Log(f"Reflection correction applied: {', '.join(reflected_axes)}-axis", tag="Eval")
    detailed_comparison = None
    # Detailed coordinate comparison table
    # detailed_comparison = create_detailed_comparison_table(
    #     traj_ref, traj_est_aligned, ape_metric.error, plot_dir, label
    # )

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    
    # Show reflection info in the title
    title = f"ATE RMSE: {ape_stat:.4f}m"
    if reflection_info['reflection_applied']:
        reflected_axes = [axis for axis in ['x', 'y', 'z'] if reflection_info[f'{axis}_reflected']]
        title += f" (Reflection corrected: {','.join(reflected_axes)}-axis)"
    ax.set_title(title)
    
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    try:
        # Try a colored trajectory via traj_colormap
        evo.tools.plot.traj_colormap(
            ax,
            traj_est_aligned,
            ape_metric.error,
            plot_mode,
            min_map=ape_stats["min"],
            max_map=ape_stats["max"],
        )
    except (ValueError, AttributeError) as e:
        # Fall back to a single-color trajectory if the colorbar fails
        Log(f"Warning: Failed to create colorbar for trajectory plot: {e}, using single color instead")
        evo.tools.plot.traj(ax, plot_mode, traj_est_aligned, "-", "blue", "est")
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)  # free memory

    return ape_stat, detailed_comparison
# def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, final = False):
#     ## Plot
#     traj_ref = PosePath3D(poses_se3=poses_gt)
#     traj_est = PosePath3D(poses_se3=poses_est)
#     print("Estimated traj length:", len(traj_est.positions_xyz))
#     print("Reference traj length:", len(traj_ref.positions_xyz))

#     traj_est_aligned = trajectory.align_trajectory(
#         traj_est, traj_ref, correct_scale=monocular
#     )   
#     ## RMSE
#     pose_relation = metrics.PoseRelation.translation_part
#     data = (traj_ref, traj_est_aligned)
#     ape_metric = metrics.APE(pose_relation)
#     ape_metric.process_data(data)
#     ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
#     ape_stats = ape_metric.get_all_statistics()
#     Log("RMSE ATE \[m]", ape_stat, tag="Eval")

#     with open(
#         os.path.join(plot_dir, "stats_{}.json".format(str(label))),
#         "w",
#         encoding="utf-8",
#     ) as f:
#         json.dump(ape_stats, f, indent=4)

#     plot_mode = evo.tools.plot.PlotMode.xy
#     fig = plt.figure()
#     ax = evo.tools.plot.prepare_axis(fig, plot_mode)
#     ax.set_title(f"ATE RMSE: {ape_stat}")
#     evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
#     evo.tools.plot.traj_colormap(
#         ax,
#         traj_est_aligned,
#         ape_metric.error,
#         plot_mode,
#         min_map=ape_stats["min"],
#         max_map=ape_stats["max"],
#     )
#     ax.legend()
#     plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

#     return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
        final=final
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        # if idx in kf_indices:
        #     continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def equirectangular_to_pinhole(
    equirect_img,
    yaw,
    pitch,
    fx,
    fy,
    cx,
    cy,
    width,
    height,
):
    device = equirect_img.device
    _, H_pano, W_pano = equirect_img.shape
    u = torch.arange(0, width, dtype=torch.float32, device=device)
    v = torch.arange(0, height, dtype=torch.float32, device=device)
    u_grid, v_grid = torch.meshgrid(u, v, indexing="xy")
    x_cam = (u_grid - cx) / fx
    y_cam = (v_grid - cy) / fy
    z_cam = torch.ones_like(x_cam)
    ray_length = torch.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_norm = x_cam / ray_length
    y_norm = y_cam / ray_length
    z_norm = z_cam / ray_length
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    x_rot1 = x_norm
    y_rot1 = y_norm * cos_pitch - z_norm * sin_pitch
    z_rot1 = y_norm * sin_pitch + z_norm * cos_pitch
    x_world = x_rot1 * cos_yaw + z_rot1 * sin_yaw
    y_world = y_rot1
    z_world = -x_rot1 * sin_yaw + z_rot1 * cos_yaw
    theta = torch.atan2(x_world, z_world)
    phi = torch.asin(torch.clamp(y_world, -1.0, 1.0))
    u_pano = (theta + np.pi) / (2 * np.pi) * W_pano
    v_pano = (np.pi / 2 - phi) / np.pi * H_pano
    u_pano_norm = (u_pano / W_pano) * 2 - 1
    v_pano_norm = (v_pano / H_pano) * 2 - 1
    grid = torch.stack([u_pano_norm, v_pano_norm], dim=-1).unsqueeze(0)
    equirect_batch = equirect_img.unsqueeze(0)
    pinhole_img = F.grid_sample(
        equirect_batch,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return pinhole_img.squeeze(0)


DEFAULT_PINHOLE_INTRINSICS = {
    "fx": 277.12812921102045,
    "fy": 415.6921938165307,
    "cx": 480.0,
    "cy": 240.0,
    "width": 960,
    "height": 480,
    "hfov_degrees": 120.0,
    "vfov_degrees": 60.0,
}


def _resolve_pinhole_intrinsics(pinhole_intrinsics):
    if pinhole_intrinsics is None:
        return DEFAULT_PINHOLE_INTRINSICS
    return {**DEFAULT_PINHOLE_INTRINSICS, **pinhole_intrinsics}


def sample_three_viewports(equirect_img, pinhole_intrinsics=None):
    intr = _resolve_pinhole_intrinsics(pinhole_intrinsics)
    viewport_width = intr["width"]
    viewport_height = intr["height"]
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    yaw_angles = [2 * np.pi * i / 3 for i in range(3)]
    pitch = 0.0
    viewports = []
    for yaw in yaw_angles:
        viewports.append(
            equirectangular_to_pinhole(
                equirect_img,
                yaw=yaw,
                pitch=pitch,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=viewport_width,
                height=viewport_height,
            )
        )
    return viewports


def eval_panoramic_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    pinhole_intrinsics=None,
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        # if idx in kf_indices:
        #     continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]
        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        pred_viewports = sample_three_viewports(image, pinhole_intrinsics)
        gt_viewports = sample_three_viewports(gt_image, pinhole_intrinsics)
        psnr_scores, ssim_scores, lpips_scores = [], [], []
        for pred_vp, gt_vp in zip(pred_viewports, gt_viewports):
            mask = gt_vp > 0
            psnr_score = psnr((pred_vp[mask]).unsqueeze(0), (gt_vp[mask]).unsqueeze(0))
            ssim_score = ssim(pred_vp.unsqueeze(0), gt_vp.unsqueeze(0))
            lpips_score = cal_lpips(pred_vp.unsqueeze(0), gt_vp.unsqueeze(0))
            psnr_scores.append(psnr_score.item())
            ssim_scores.append(ssim_score.item())
            lpips_scores.append(lpips_score.item())

        psnr_array.append(float(np.mean(psnr_scores)))
        ssim_array.append(float(np.mean(ssim_scores)))
        lpips_array.append(float(np.mean(lpips_scores)))
    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )
    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)
    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion (qx, qy, qz, qw).
    
    Args:
        R: 3x3 rotation matrix (numpy array or torch tensor)
    
    Returns:
        q: quaternion [qx, qy, qz, qw]
    """
    if torch.is_tensor(R):
        R = R.cpu().numpy()
    
    # Ensure R is a valid rotation matrix
    R = np.array(R)
    
    # Compute the quaternion
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qx, qy, qz, qw])


def save_trajectory_tum_format(frames, save_dir, filename="trajectory.txt"):
    """
    Save the full-frame trajectory in TUM format.
    
    TUM format: timestamp tx ty tz qx qy qz qw
    
    Args:
        frames: dict of all frame cameras
        save_dir: output directory
        filename: output filename
    """
    mkdir_p(save_dir)
    tum_path = os.path.join(save_dir, filename)
    
    Log(f"Saving TUM trajectory to: {tum_path}", tag="Save")
    
    # Collect poses sorted by frame id
    frame_ids = sorted(frames.keys())
    
    with open(tum_path, 'w') as f:
        f.write("# TUM trajectory format\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        for frame_id in frame_ids:
            frame = frames[frame_id]
            
            # Pose (world-to-camera)
            R = frame.R.cpu().numpy() if torch.is_tensor(frame.R) else frame.R
            T = frame.T.cpu().numpy() if torch.is_tensor(frame.T) else frame.T
            
            # Build the camera-to-world matrix
            pose_c2w = np.eye(4)
            pose_c2w[:3, :3] = R.T  # R^T
            pose_c2w[:3, 3] = -R.T @ T  # camera center
            
            # Extract position and rotation
            position = pose_c2w[:3, 3]
            rotation = pose_c2w[:3, :3]
            
            # Convert to quaternion
            quaternion = rotation_matrix_to_quaternion(rotation)
            
            # Use the frame timestamp if present, else the frame id
            timestamp = frame.timestamp if hasattr(frame, 'timestamp') else float(frame_id)
            
            # TUM row: timestamp tx ty tz qx qy qz qw
            f.write(f"{timestamp:.6f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                   f"{quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")
    
    Log(f"TUM trajectory saved ({len(frame_ids)} frames)", tag="Save")
    return tum_path


def save_trajectory_tum_format_keyframes_only(frames, kf_ids, save_dir, filename="trajectory_keyframes.txt"):
    """
    Save the keyframe trajectory in TUM format.
    
    Args:
        frames: dict of all frame cameras
        kf_ids: keyframe ids
        save_dir: output directory
        filename: output filename
    """
    mkdir_p(save_dir)
    tum_path = os.path.join(save_dir, filename)
    
    Log(f"Saving keyframe TUM trajectory to: {tum_path}", tag="Save")
    
    with open(tum_path, 'w') as f:
        f.write("# TUM trajectory format (keyframes only)\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        
        for frame_id in kf_ids:
            frame = frames[frame_id]
            
            # Get the pose
            R = frame.R.cpu().numpy() if torch.is_tensor(frame.R) else frame.R
            T = frame.T.cpu().numpy() if torch.is_tensor(frame.T) else frame.T
            
            # Build the camera-to-world matrix
            pose_c2w = np.eye(4)
            pose_c2w[:3, :3] = R.T
            pose_c2w[:3, 3] = -R.T @ T
            
            # Extract position and rotation
            position = pose_c2w[:3, 3]
            rotation = pose_c2w[:3, :3]
            
            # Convert to quaternion
            quaternion = rotation_matrix_to_quaternion(rotation)
            
            # Use the frame timestamp if present, else the frame id
            timestamp = frame.timestamp if hasattr(frame, 'timestamp') else float(frame_id)
            
            # TUM row
            f.write(f"{timestamp:.6f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                   f"{quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}\n")
    
    Log(f"Keyframe TUM trajectory saved ({len(kf_ids)} frames)", tag="Save")
    return tum_path
