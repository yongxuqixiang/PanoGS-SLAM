"""
Standalone ATE evaluation with evo.

Compares a saved estimated trajectory (trj_XXXX.json produced by slam.py)
against the dataset ground truth (data_extrinsics_sorted.json +
data_views_sorted.json), with Sim(3) alignment for monocular SLAM.

Usage:
    python tools/eval.py \
        --extrinsics datasets/<scene>/data_extrinsics_sorted.json \
        --views      datasets/<scene>/data_views_sorted.json \
        --traj       results/<run>/plot/trj_XXXX.json
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core import metrics, trajectory
from evo.tools import plot


def load_ground_truth_poses(extrinsics_path, views_path):
    """
    Load the ground-truth trajectory, mirroring PanoramicDataset in dataset.py.

    Args:
        extrinsics_path: path to data_extrinsics_sorted.json
        views_path: path to data_views_sorted.json (defines the frame order)

    Returns:
        (list of pose_ids, list of 4x4 C2W numpy matrices)
    """
    print(f"Loading ground truth from {extrinsics_path} and {views_path}...")

    with open(extrinsics_path, "r") as f:
        extrinsics_data = json.load(f)
    with open(views_path, "r") as f:
        views_data = json.load(f)

    extrinsics_dict = {}
    for extrinsic in extrinsics_data["extrinsics"]:
        key = extrinsic["key"]
        rotation = np.array(extrinsic["value"]["rotation"])
        center = np.array(extrinsic["value"]["center"])
        extrinsics_dict[key] = (rotation, center)

    poses_gt_c2w = []
    pose_ids = []
    first_center = None

    for view in views_data["views"]:
        view_data = view["value"]["ptr_wrapper"]["data"]
        id_pose = view_data["id_pose"]
        pose_ids.append(id_pose)

        rotation, center = extrinsics_dict[id_pose]

        # Normalize w.r.t. the first camera center (as in dataset.py)
        if first_center is None:
            first_center = center.copy()

        center_normalized = center - first_center

        # World-to-Camera (W2C) -> invert for evaluation
        pose_w2c = np.eye(4)
        pose_w2c[:3, :3] = rotation
        pose_w2c[:3, 3] = -rotation @ center_normalized
        poses_gt_c2w.append(np.linalg.inv(pose_w2c))

    print(f"Loaded {len(poses_gt_c2w)} ground-truth poses.")
    return pose_ids, poses_gt_c2w


def load_estimated_poses(trj_json_path):
    """
    Load the estimated trajectory from a trj_XXXX.json file.

    Returns:
        (list of pose_ids, list of 4x4 C2W numpy matrices)
    """
    print(f"Loading estimated trajectory from {trj_json_path}...")
    with open(trj_json_path, "r") as f:
        trj_data = json.load(f)

    # trj.json stores Camera-to-World (C2W) matrices directly
    poses_est_c2w = [np.array(p) for p in trj_data["trj_est"]]
    pose_ids = trj_data["trj_id"]

    print(f"Loaded {len(poses_est_c2w)} estimated poses.")
    return pose_ids, poses_est_c2w


def evaluate_trajectory(poses_gt, poses_est, monocular=True, output="trajectory_comparison.png"):
    """
    Align the two trajectories with evo and report ATE statistics.

    Args:
        poses_gt: list of ground-truth C2W matrices
        poses_est: list of estimated C2W matrices
        monocular: if True, align with Sim(3) (scale-corrected);
            set False for stereo/RGB-D to use SE(3) alignment
        output: path of the saved trajectory plot
    """
    traj_ref = trajectory.PosePath3D(poses_se3=poses_gt)
    traj_est = trajectory.PosePath3D(poses_se3=poses_est)

    print(f"Aligning trajectories (scale correction: {monocular})...")
    traj_est_aligned = trajectory.align_trajectory(traj_est, traj_ref, correct_scale=monocular)

    # Absolute pose error (APE) on the translation part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)

    ape_stats = ape_metric.get_all_statistics()
    ate_rmse = ape_stats["rmse"]

    print("\n--- Trajectory evaluation (ATE) ---")
    print(f"  RMSE:   {ate_rmse:.4f} (m)")
    print(f"  Mean:   {ape_stats['mean']:.4f} (m)")
    print(f"  Median: {ape_stats['median']:.4f} (m)")
    print(f"  Std:    {ape_stats['std']:.4f} (m)")
    print(f"  Min:    {ape_stats['min']:.4f} (m)")
    print(f"  Max:    {ape_stats['max']:.4f} (m)")
    print("------------------------------------\n")

    # Trajectory plot
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ate_rmse:.4f} m")

    plot.traj(ax, plot_mode, traj_ref, "--", "gray", "Ground Truth")
    plot.traj_colormap(
        ax, traj_est_aligned, ape_metric.error, plot_mode,
        min_map=ape_stats["min"], max_map=ape_stats["max"]
    )
    ax.legend()

    plt.savefig(output, dpi=150)
    print(f"Trajectory comparison saved to: {output}")
    return ate_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extrinsics", required=True,
                        help="path to data_extrinsics_sorted.json of the scene")
    parser.add_argument("--views", required=True,
                        help="path to data_views_sorted.json of the scene")
    parser.add_argument("--traj", required=True,
                        help="path to the estimated trj_XXXX.json saved by slam.py")
    parser.add_argument("--output", default="trajectory_comparison.png",
                        help="output path of the trajectory plot")
    parser.add_argument("--start-index", type=int, default=0,
                        help="skip the first N keyframes of the estimated trajectory")
    parser.add_argument("--no-scale", action="store_true",
                        help="use SE(3) alignment instead of Sim(3) (stereo/RGB-D)")
    args = parser.parse_args()

    gt_ids, gt_poses = load_ground_truth_poses(args.extrinsics, args.views)
    est_ids, est_poses = load_estimated_poses(args.traj)

    # Match the GT poses to the estimated trajectory order
    gt_poses_dict = {id: pose for id, pose in zip(gt_ids, gt_poses)}

    aligned_gt_poses = []
    valid_est_poses = []

    for i, est_id in enumerate(est_ids[args.start_index:], start=args.start_index):
        if est_id in gt_poses_dict:
            aligned_gt_poses.append(gt_poses_dict[est_id])
            valid_est_poses.append(est_poses[i])
        else:
            print(f"Warning: no ground-truth pose with id {est_id}, skipping.")

    if len(aligned_gt_poses) < 2:
        raise SystemExit("Error: fewer than 2 matched poses, cannot evaluate.")

    print(f"\nMatched {len(aligned_gt_poses)} poses, evaluating...")
    evaluate_trajectory(aligned_gt_poses, valid_est_poses,
                         monocular=not args.no_scale, output=args.output)
