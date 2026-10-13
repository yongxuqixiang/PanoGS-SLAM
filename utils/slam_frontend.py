import time
import json
import os
from tracemalloc import start
from collections import defaultdict

import numpy as np  # pyright: ignore[reportMissingImports]
import torch  # pyright: ignore[reportMissingImports]
import torch.multiprocessing as mp  # pyright: ignore[reportMissingImports]
import torch.nn.functional as F  # pyright: ignore[reportMissingImports]

try:
    import cv2  # pyright: ignore[reportMissingImports]
    from PIL import Image
    HAS_VISUALIZATION_DEPS = True
except ImportError:
    HAS_VISUALIZATION_DEPS = False
    print("Warning: cv2 or PIL not available, depth visualization will be disabled")

from gaussian_splatting.gaussian_renderer import render
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import (
    get_loss_tracking,
    get_median_depth,
)

class Profiler:
    """Profiler recording per-operation timings."""
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_section = None
        self.start_time = None
        
    def start(self, section_name):
        """Start timing an operation."""
        if self.start_time is not None:
            # Close any unfinished timing first
            self.end()
        self.current_section = section_name
        self.start_time = time.time()
        
    def end(self):
        """End the current timing."""
        if self.start_time is not None and self.current_section is not None:
            elapsed = time.time() - self.start_time
            self.timings[self.current_section].append(elapsed)
            self.current_section = None
            self.start_time = None
            
    def record(self, section_name, elapsed_time):
        """Record an operation duration directly."""
        self.timings[section_name].append(elapsed_time)
        
    def print_stats(self, frame_idx=None):
        """Print the statistics."""
        if frame_idx is not None:
            print(f"\n=== Frame {frame_idx} profiling ===")
        else:
            print(f"\n=== Profiling ===")
        
        total_time = 0
        for section, times in sorted(self.timings.items()):
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                total = sum(times)
                total_time += total
                print(f"  {section:30s}: avg={avg_time*1000:6.2f}ms, "
                      f"max={max_time*1000:6.2f}ms, "
                      f"min={min_time*1000:6.2f}ms, total={total*1000:6.2f}ms, n={len(times)}")
        print(f"  total: {total_time*1000:.2f}ms")
        print("=" * 50)
        
    def clear(self):
        """Clear all records."""
        self.timings.clear()
        
    def get_total(self):
        """Return the total time."""
        return sum(sum(times) for times in self.timings.values())
try:
    from utils.depth_estimation import create_depth_estimator
except ImportError:
    create_depth_estimator = None


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1
        
        # Profiler
        self.profiler = Profiler()

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.last_render_image = None  # kept for visualization
        self.last_render_opacity = None  # for opacity-guided sampling
        
        # Motion model: velocities maintained by a first-order filter
        self.velocity_T = None  # translation velocity
        self.velocity_R = None  # rotation velocity (lie algebra)
        self.filter_alpha = config.get("Training", {}).get("velocity_filter_alpha", 0.2)  # filter coefficient
        
        # Depth estimator
        self.depth_estimator = None
        self.use_depth_estimation = config.get("depth_estimation", {}).get("enabled", False)

        # Visualization config
        self.save_depth_visualization = config.get("depth_estimation", {}).get("save_visualization", True)
        self.visualization_dir = os.path.join(config.get("Results", {}).get("save_dir", "results"), "depth_visualization")

        # Rendered-image saving config
        self.save_render_images = config.get("Results", {}).get("save_render_images", True)
        self.render_images_dir = os.path.join(config.get("Results", {}).get("save_dir", "results"), "render_images")
        self.render_images_interval = config.get("Results", {}).get("render_images_interval", 100)  # save every 100 frames
        self.save_render_depth = config.get("Results", {}).get("save_render_depth", True)
        self.render_depth_dir = os.path.join(config.get("Results", {}).get("save_dir", "results"), "render_depth")
        
        # Optional: fixed-viewpoint pinhole renders with the trajectory
        self.fixed_pinhole_render_cfg = config.get("Results", {}).get("fixed_pinhole_render", {})
        self.enable_fixed_pinhole_render = bool(self.fixed_pinhole_render_cfg.get("enabled", False))
        self.fixed_pinhole_render_interval = int(
            self.fixed_pinhole_render_cfg.get("interval", self.render_images_interval)
        )
        self.fixed_pinhole_render_dir = os.path.join(
            config.get("Results", {}).get("save_dir", "results"),
            self.fixed_pinhole_render_cfg.get("save_dirname", "fixed_pinhole_render"),
        )
        self.fixed_pinhole_draw_est = bool(self.fixed_pinhole_render_cfg.get("draw_estimated_traj", True))
        self.fixed_pinhole_draw_gt = bool(self.fixed_pinhole_render_cfg.get("draw_gt_traj", True))
        self.fixed_pinhole_camera = None
        self.fixed_pinhole_R = None
        self.fixed_pinhole_t = None
        self.fixed_pinhole_intrinsics = {}
        
        # Edge-loss visualization config
        self.save_edge_loss_visualization = config.get("Training", {}).get("save_edge_loss_visualization", False)
        self.edge_loss_visualization_dir = os.path.join(config.get("Results", {}).get("save_dir", "results"), "edge_loss_visualization")
        self.edge_loss_visualization_interval = config.get("Training", {}).get("edge_loss_visualization_interval", 10)  # save every 10 frames
        
    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        
        # First frame to process (default 0)
        self.start_frame = self.config["Dataset"].get("start_frame", 0)
        
        # Initialize the BiFuse depth estimator
        if self.use_depth_estimation:
            try:
                if create_depth_estimator is not None:
                    self.depth_estimator = create_depth_estimator(self.config)
                    model_type = self.config.get('depth_estimation', {}).get('model', 'bifuse')
                    pretrained_path = self.config.get('depth_estimation', {}).get('pretrained_path', 'default')
                    Log(f"Depth estimator enabled: weights={pretrained_path}")
                else:
                    Log("Depth estimation module unavailable, falling back to random depth")
                    self.use_depth_estimation = False
            except Exception as e:
                Log(f"Failed to initialize depth estimator: {e}, falling back to random depth")
                self.use_depth_estimation = False
        
        # Create the visualization directory
        if self.save_depth_visualization:
            os.makedirs(self.visualization_dir, exist_ok=True)
            Log(f"Depth visualization will be saved to: {self.visualization_dir}")
        
        # Create the rendered-image directory
        if self.save_render_images:
            os.makedirs(self.render_images_dir, exist_ok=True)
            Log(f"Render images will be saved to: {self.render_images_dir} (every {self.render_images_interval} frames)")
        
        # Create the rendered-depth directory
        if self.save_render_depth:
            os.makedirs(self.render_depth_dir, exist_ok=True)
            Log(f"Render depth will be saved to: {self.render_depth_dir} (every {self.render_images_interval} frames)")
        
        # Fixed-viewpoint pinhole render config
        if self.fixed_pinhole_render_interval <= 0:
            self.fixed_pinhole_render_interval = max(1, self.render_images_interval)
        pinhole_intr = self.config.get("Results", {}).get("pinhole_intrinsics", {})
        self.fixed_pinhole_intrinsics = {
            "width": int(self.fixed_pinhole_render_cfg.get("width", pinhole_intr.get("width", self.dataset.width))),
            "height": int(self.fixed_pinhole_render_cfg.get("height", pinhole_intr.get("height", self.dataset.height))),
            "fx": float(self.fixed_pinhole_render_cfg.get("fx", pinhole_intr.get("fx", self.dataset.fx))),
            "fy": float(self.fixed_pinhole_render_cfg.get("fy", pinhole_intr.get("fy", self.dataset.fy))),
            "cx": float(self.fixed_pinhole_render_cfg.get("cx", pinhole_intr.get("cx", self.dataset.cx))),
            "cy": float(self.fixed_pinhole_render_cfg.get("cy", pinhole_intr.get("cy", self.dataset.cy))),
            "pose_tum": self.fixed_pinhole_render_cfg.get("pose_tum", None),
            "cam_pos": np.asarray(self.fixed_pinhole_render_cfg.get("cam_pos", [-2.2, 0.3, 0.95]), dtype=np.float32),
            "cam_target": np.asarray(self.fixed_pinhole_render_cfg.get("cam_target", [-4.5, 0.6, 2.0]), dtype=np.float32),
            "cam_up": np.asarray(self.fixed_pinhole_render_cfg.get("cam_up", [0.0, 0.0, 1.0]), dtype=np.float32),
            "znear": float(self.fixed_pinhole_render_cfg.get("znear", 0.05)),
            "zfar": float(self.fixed_pinhole_render_cfg.get("zfar", 50.0)),
        }
        if self.enable_fixed_pinhole_render:
            os.makedirs(self.fixed_pinhole_render_dir, exist_ok=True)
            Log(
                f"Fixed pinhole render enabled: {self.fixed_pinhole_render_dir} "
                f"(every {self.fixed_pinhole_render_interval} frames)"
            )
        
        # Create the edge-loss visualization directory
        if self.save_edge_loss_visualization:
            os.makedirs(self.edge_loss_visualization_dir, exist_ok=True)
            Log(f"Edge loss visualization will be saved to: {self.edge_loss_visualization_dir} (every {self.edge_loss_visualization_interval} frames)")

    def save_depth_visualization_data(self, cur_frame_idx, gt_img, estimated_depth, initial_depth, valid_rgb):
        """Save depth-estimation visualization results."""
        if not self.save_depth_visualization or not HAS_VISUALIZATION_DEPS:
            return
        
        try:
            # Create the frame subdirectory
            frame_dir = os.path.join(self.visualization_dir, f"frame_{cur_frame_idx:06d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            # Save the original image
            gt_img_np = gt_img.cpu().numpy().transpose(1, 2, 0)
            gt_img_uint8 = (np.clip(gt_img_np, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(gt_img_uint8).save(os.path.join(frame_dir, "original_image.png"))
            
            # Save the valid-pixel mask
            valid_mask_np = valid_rgb.cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(valid_mask_np, mode='L').save(os.path.join(frame_dir, "valid_mask.png"))
            
            # Save the estimated depth (normalized to 0-255)
            if estimated_depth is not None:
                depth_np = estimated_depth.cpu().numpy()
                depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)).save(
                    os.path.join(frame_dir, "estimated_depth_colored.png")
                )
                
                # Save the raw depth values (for analysis)
                np.save(os.path.join(frame_dir, "estimated_depth.npy"), depth_np)
                
                # Save depth statistics
                depth_stats = {
                    "min": float(depth_np.min()),
                    "max": float(depth_np.max()),
                    "mean": float(depth_np.mean()),
                    "std": float(depth_np.std()),
                    "median": float(np.median(depth_np))
                }
                with open(os.path.join(frame_dir, "depth_stats.json"), 'w') as f:
                    json.dump(depth_stats, f, indent=2)
            
            # Save the final depth (scaled)
            if initial_depth is not None:
                final_depth_np = initial_depth.cpu().numpy()
                if final_depth_np.ndim == 3:
                    final_depth_np = final_depth_np[0]
                
                final_depth_normalized = (final_depth_np - final_depth_np.min()) / (final_depth_np.max() - final_depth_np.min() + 1e-8)
                final_depth_colored = cv2.applyColorMap((final_depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                Image.fromarray(cv2.cvtColor(final_depth_colored, cv2.COLOR_BGR2RGB)).save(
                    os.path.join(frame_dir, "final_depth_colored.png")
                )
                
                # Save the final depth values
                np.save(os.path.join(frame_dir, "final_depth.npy"), final_depth_np)
            
            # Save the comparison image (image + depth)
            if estimated_depth is not None and initial_depth is not None:
                self._save_comparison_image(frame_dir, gt_img_np, estimated_depth.cpu().numpy(), 
                                          initial_depth.cpu().numpy(), valid_rgb.cpu().numpy())
            
            Log(f"Depth visualization saved for frame {cur_frame_idx}")
            
        except Exception as e:
            Log(f"Failed to save depth visualization for frame {cur_frame_idx}: {e}")
    
    def _save_comparison_image(self, frame_dir, gt_img, estimated_depth, final_depth, valid_mask):
        """Save a comparison image."""
        try:
            # Build a 2x2 comparison
            fig_height, fig_width = gt_img.shape[:2]
            comparison = np.zeros((fig_height * 2, fig_width * 2, 3), dtype=np.uint8)
            
            # top-left: original
            comparison[:fig_height, :fig_width] = (gt_img * 255).astype(np.uint8)
            
            # top-right: estimated depth (colored)
            if estimated_depth.ndim == 3:
                estimated_depth = estimated_depth[0]
            depth_normalized = (estimated_depth - estimated_depth.min()) / (estimated_depth.max() - estimated_depth.min() + 1e-8)
            depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            comparison[:fig_height, fig_width:] = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            
            # bottom-left: final depth (colored)
            if final_depth.ndim == 3:
                final_depth = final_depth[0]
            final_depth_normalized = (final_depth - final_depth.min()) / (final_depth.max() - final_depth.min() + 1e-8)
            final_depth_colored = cv2.applyColorMap((final_depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            comparison[fig_height:, :fig_width] = cv2.cvtColor(final_depth_colored, cv2.COLOR_BGR2RGB)
            
            # bottom-right: valid mask
            valid_mask_3ch = np.stack([valid_mask * 255] * 3, axis=-1)
            comparison[fig_height:, fig_width:] = valid_mask_3ch.astype(np.uint8)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Estimated Depth", (fig_width + 10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Final Depth", (10, fig_height + 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Valid Mask", (fig_width + 10, fig_height + 30), font, 1, (255, 255, 255), 2)
            
            Image.fromarray(comparison).save(os.path.join(frame_dir, "comparison.png"))
            
        except Exception as e:
            Log(f"Failed to save comparison image: {e}")
    
    def save_edge_loss_visualization_data(self, cur_frame_idx, gt_image, render_image, 
                                          gt_edge_mask, render_edge_mask, distance_field=None):
        """Save edge-loss visualization results."""
        if not self.save_edge_loss_visualization or not HAS_VISUALIZATION_DEPS:
            return
        
        try:
            # Create the frame subdirectory
            frame_dir = os.path.join(self.edge_loss_visualization_dir, f"frame_{cur_frame_idx:06d}")
            os.makedirs(frame_dir, exist_ok=True)
            
            # Convert to numpy
            gt_img_np = gt_image.detach().cpu().numpy().transpose(1, 2, 0)
            gt_img_uint8 = (np.clip(gt_img_np, 0, 1) * 255).astype(np.uint8)
            
            render_img_np = render_image.detach().cpu().numpy().transpose(1, 2, 0)
            render_img_uint8 = (np.clip(render_img_np, 0, 1) * 255).astype(np.uint8)
            
            # Save the original image
            Image.fromarray(gt_img_uint8).save(os.path.join(frame_dir, "01_gt_image.png"))
            Image.fromarray(render_img_uint8).save(os.path.join(frame_dir, "02_render_image.png"))
            
            # Save the GT edge mask
            gt_edge_np = gt_edge_mask.detach().cpu().numpy()
            if gt_edge_np.ndim == 3:
                gt_edge_np = gt_edge_np[0]
            gt_edge_uint8 = (np.clip(gt_edge_np, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(gt_edge_uint8, mode='L').save(os.path.join(frame_dir, "03_gt_edge_mask.png"))
            
            # Save the rendered edge mask
            render_edge_np = render_edge_mask.detach().cpu().numpy()
            if render_edge_np.ndim == 3:
                render_edge_np = render_edge_np[0]
            # Normalize to 0-255
            render_edge_normalized = (render_edge_np - render_edge_np.min()) / (render_edge_np.max() - render_edge_np.min() + 1e-8)
            render_edge_uint8 = (render_edge_normalized * 255).astype(np.uint8)
            Image.fromarray(render_edge_uint8, mode='L').save(os.path.join(frame_dir, "04_render_edge_mask.png"))
            
            # Save the distance field if present
            if distance_field is not None:
                dist_field_np = distance_field.detach().cpu().numpy()
                if dist_field_np.ndim == 3:
                    dist_field_np = dist_field_np[0]
                
                # Visualize the distance field with the jet colormap
                dist_field_colored = cv2.applyColorMap((dist_field_np / dist_field_np.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)
                Image.fromarray(cv2.cvtColor(dist_field_colored, cv2.COLOR_BGR2RGB)).save(
                    os.path.join(frame_dir, "05_distance_field.png")
                )
                
                # Save the raw distance field
                np.save(os.path.join(frame_dir, "distance_field.npy"), dist_field_np)
            
            # Build the comparison (3x2 layout)
            self._save_edge_loss_comparison_image(
                frame_dir, gt_img_uint8, render_img_uint8, 
                gt_edge_np, render_edge_np, 
                distance_field.detach().cpu().numpy()[0] if distance_field is not None else None
            )
            
            Log(f"Edge loss visualization saved for frame {cur_frame_idx}")
            
        except Exception as e:
            Log(f"Failed to save edge loss visualization for frame {cur_frame_idx}: {e}")
    
    def _save_edge_loss_comparison_image(self, frame_dir, gt_img, render_img, 
                                         gt_edge, render_edge, distance_field=None):
        """Save an edge-loss comparison image."""
        try:
            h, w = gt_img.shape[:2]
            
            # Build the 3x2 layout
            if distance_field is not None:
                comparison = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
                n_cols = 3
            else:
                comparison = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
                n_cols = 2
            
            # Row 1: original and rendered
            comparison[:h, :w] = gt_img
            comparison[:h, w:2*w] = render_img
            
            # Row 2: GT edges and rendered edges
            gt_edge_3ch = np.stack([gt_edge * 255] * 3, axis=-1).astype(np.uint8)
            comparison[h:, :w] = gt_edge_3ch
            
            render_edge_normalized = (render_edge - render_edge.min()) / (render_edge.max() - render_edge.min() + 1e-8)
            render_edge_3ch = np.stack([render_edge_normalized * 255] * 3, axis=-1).astype(np.uint8)
            comparison[h:, w:2*w] = render_edge_3ch
            
            # Column 3: distance field if present
            if distance_field is not None and n_cols == 3:
                # Top: distance field
                dist_field_colored = cv2.applyColorMap(
                    (distance_field / distance_field.max() * 255).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                comparison[:h, 2*w:] = cv2.cvtColor(dist_field_colored, cv2.COLOR_BGR2RGB)
                
                # Bottom: edge overlay comparison
                overlay = gt_img.copy()
                # GT edges: green
                overlay[gt_edge > 0.5] = [0, 255, 0]
                # Rendered edges: red
                render_edge_binary = render_edge_normalized > 0.5
                overlay[render_edge_binary] = [255, 0, 0]
                # Overlap: yellow
                overlap = (gt_edge > 0.5) & render_edge_binary
                overlay[overlap] = [255, 255, 0]
                comparison[h:, 2*w:] = overlay
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            color = (255, 255, 255)
            
            cv2.putText(comparison, "GT Image", (10, 30), font, font_scale, color, thickness)
            cv2.putText(comparison, "Render Image", (w + 10, 30), font, font_scale, color, thickness)
            cv2.putText(comparison, "GT Edge", (10, h + 30), font, font_scale, color, thickness)
            cv2.putText(comparison, "Render Edge", (w + 10, h + 30), font, font_scale, color, thickness)
            
            if n_cols == 3:
                cv2.putText(comparison, "Distance Field", (2*w + 10, 30), font, font_scale, color, thickness)
                cv2.putText(comparison, "Edge Overlay", (2*w + 10, h + 30), font, font_scale, color, thickness)
                # Add the legend
                cv2.putText(comparison, "Green=GT", (2*w + 10, h + 60), font, 0.6, (0, 255, 0), 2)
                cv2.putText(comparison, "Red=Render", (2*w + 10, h + 85), font, 0.6, (255, 0, 0), 2)
                cv2.putText(comparison, "Yellow=Overlap", (2*w + 10, h + 110), font, 0.6, (255, 255, 0), 2)
            
            Image.fromarray(comparison).save(os.path.join(frame_dir, "00_comparison.png"))
            
        except Exception as e:
            Log(f"Failed to save edge loss comparison image: {e}")
    
    def _create_panoramic_depth_init(self, gt_img):
        """
        Create a smoother depth initialization for panoramic cameras,
        reducing numerical errors in spherical coordinate conversion.
        """
        height, width = gt_img.shape[1], gt_img.shape[2]
        
        # Build the spherical coordinate grid
        theta = torch.linspace(-torch.pi, torch.pi, width, device=gt_img.device)
        phi = torch.linspace(torch.pi/2, -torch.pi/2, height, device=gt_img.device)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='xy')
        
        # Smooth depth distribution over the sphere
        # More conservative depths near the poles
        depth_base = 2.0 + 0.5 * torch.cos(phi_grid)  # smaller depth near the poles
        
        # Add image-content-based depth variation
        if gt_img.shape[0] == 3:  # RGB image
            gray = torch.mean(gt_img, dim=0)
            # Brightness as a depth prior (smoothed)
            depth_variation = 0.3 * (1.0 - gray)  # darker regions get larger depth
        else:
            depth_variation = torch.zeros_like(depth_base)
        
        # Smooth random noise to avoid sharp transitions in spherical coords
        noise = torch.randn_like(depth_base) * 0.1
        
        # Smaller noise near the ±pi seam
        boundary_mask = torch.abs(theta_grid) > 0.8 * torch.pi
        boundary_reduction = self.config["Training"].get("panoramic_boundary_noise_reduction", 0.3)
        noise[boundary_mask] *= boundary_reduction
        
        initial_depth = depth_base + depth_variation + noise
        
        # Keep depths within a reasonable range
        initial_depth = torch.clamp(initial_depth, 0.5, 5.0)
        
        return initial_depth.unsqueeze(0)
    
    def _fill_invalid_depth_with_nearest(self, depth_2d, valid_mask_2d, fallback_depth):
            """
            Fill invalid depth pixels with the nearest valid depth.
            Falls back to the raw depth map if no valid pixel exists.
            """
            if valid_mask_2d.all():
                return depth_2d

            if not valid_mask_2d.any():
                return torch.full_like(depth_2d, fallback_depth)

            if "cv2" not in globals():
                filled = depth_2d.clone()
                filled[~valid_mask_2d] = fallback_depth
                return filled

            depth_np = depth_2d.detach().cpu().numpy()
            valid_np = valid_mask_2d.detach().cpu().numpy().astype(np.uint8)
            # distanceTransformWithLabels computes, for non-zero pixels, the distance
            # to the nearest zero pixel; valid=0 / invalid=1 gives the nearest valid label.
            src = 1 - valid_np

            try:
                _, labels = cv2.distanceTransformWithLabels(
                    src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
                )
            except Exception:
                return depth_2d

            valid_coords = np.column_stack(np.where(valid_np > 0))
            if valid_coords.shape[0] == 0:
                filled = depth_2d.clone()
                filled[~valid_mask_2d] = fallback_depth
                return filled

            labels = labels.astype(np.int64) - 1
            labels = np.clip(labels, 0, valid_coords.shape[0] - 1)
            nearest_coords = valid_coords[labels]

            filled_np = depth_np.copy()
            invalid_np = valid_np == 0
            nearest_depth = depth_np[nearest_coords[..., 0], nearest_coords[..., 1]]
            filled_np[invalid_np] = nearest_depth[invalid_np]

            return torch.from_numpy(filled_np).to(device=depth_2d.device, dtype=depth_2d.dtype)

    @staticmethod
    def _quaternion_to_rotation_matrix(quat):
        qx, qy, qz, qw = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm([qx, qy, qz, qw])
        if norm < 1e-12:
            raise ValueError("fixed_pinhole_render.pose_tum quaternion norm too small")
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=np.float64,
        )

    def _parse_fixed_pinhole_pose_tum(self, pose_tum):
        if pose_tum is None:
            return None
        if isinstance(pose_tum, str):
            tokens = [x for x in pose_tum.strip().split() if x]
        else:
            try:
                tokens = list(pose_tum)
            except TypeError as exc:
                raise ValueError("fixed_pinhole_render.pose_tum must be a string or list of numbers") from exc

        if len(tokens) == 7:
            tx, ty, tz, qx, qy, qz, qw = map(float, tokens)
        elif len(tokens) == 8:
            _, tx, ty, tz, qx, qy, qz, qw = map(float, tokens)
        else:
            raise ValueError("fixed_pinhole_render.pose_tum must be a 7- or 8-column TUM pose")

        R_c2w = self._quaternion_to_rotation_matrix([qx, qy, qz, qw])
        t_c2w = np.array([tx, ty, tz], dtype=np.float64)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        return R_w2c.astype(np.float32), t_w2c.astype(np.float32)

    def _build_fixed_pinhole_camera(self):
        if self.fixed_pinhole_camera is not None:
            return self.fixed_pinhole_camera

        intr = self.fixed_pinhole_intrinsics
        width = int(intr["width"])
        height = int(intr["height"])
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
        parsed_pose = self._parse_fixed_pinhole_pose_tum(intr.get("pose_tum"))
        if parsed_pose is not None:
            R, t = parsed_pose
        else:
            cam_pos = intr["cam_pos"].astype(np.float64)
            cam_target = intr["cam_target"].astype(np.float64)
            cam_up = intr["cam_up"].astype(np.float64)

            z_axis = cam_target - cam_pos
            z_norm = np.linalg.norm(z_axis)
            if z_norm < 1e-8:
                raise ValueError("fixed_pinhole_render.cam_pos and cam_target must differ")
            z_axis /= z_norm
            x_axis = np.cross(z_axis, cam_up)
            if np.linalg.norm(x_axis) < 1e-8:
                cam_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                x_axis = np.cross(z_axis, cam_up)
            x_axis /= np.linalg.norm(x_axis) + 1e-12
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis) + 1e-12

            R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)
            t = (-R @ cam_pos.astype(np.float32)).astype(np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        znear = float(intr["znear"])
        zfar = float(intr["zfar"])
        tanfovx = width / (2.0 * fx)
        tanfovy = height / (2.0 * fy)
        top = tanfovy * znear
        bottom = -top
        right = tanfovx * znear
        left = -right

        Rt = torch.from_numpy(w2c).to(device=self.device, dtype=torch.float32)
        viewmatrix = Rt.transpose(0, 1)
        cam_center = torch.linalg.inv(Rt)[:3, 3].clone()

        P = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = 1.0
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        projmatrix_raw = P.transpose(0, 1)
        projmatrix = viewmatrix @ projmatrix_raw

        self.fixed_pinhole_camera = {
            "image_width": width,
            "image_height": height,
            "tanfovx": float(tanfovx),
            "tanfovy": float(tanfovy),
            "viewmatrix": viewmatrix,
            "projmatrix": projmatrix,
            "projmatrix_raw": projmatrix_raw,
            "campos": cam_center,
        }
        self.fixed_pinhole_R = R.astype(np.float64)
        self.fixed_pinhole_t = t.astype(np.float64)
        return self.fixed_pinhole_camera

    def _project_world_points_to_fixed_camera(self, points_xyz):
        if points_xyz.shape[0] == 0:
            return np.empty((0, 2), dtype=np.int32)
        R = self.fixed_pinhole_R
        t = self.fixed_pinhole_t
        intr = self.fixed_pinhole_intrinsics
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        width, height = intr["width"], intr["height"]

        cam_pts = (R @ points_xyz.T).T + t[None, :]
        valid = cam_pts[:, 2] > 1e-4
        u = np.full((points_xyz.shape[0],), -1.0, dtype=np.float64)
        v = np.full((points_xyz.shape[0],), -1.0, dtype=np.float64)
        u[valid] = fx * cam_pts[valid, 0] / cam_pts[valid, 2] + cx
        v[valid] = fy * cam_pts[valid, 1] / cam_pts[valid, 2] + cy
        in_frame = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        uv = np.stack([u[in_frame], v[in_frame]], axis=1)
        return np.round(uv).astype(np.int32)

    def _collect_est_gt_trajectory_until(self, cur_frame_idx):
        est_pts = []
        gt_pts = []
        frame_ids = [idx for idx in sorted(self.cameras.keys()) if idx <= cur_frame_idx]
        for idx in frame_ids:
            cam = self.cameras[idx]
            if cam is None:
                continue
            est_R = cam.R.detach().cpu().numpy()
            est_t = cam.T.detach().cpu().numpy()
            gt_R = cam.R_gt.detach().cpu().numpy()
            gt_t = cam.T_gt.detach().cpu().numpy()
            est_center = (-est_R.T @ est_t).astype(np.float64)
            gt_center = (-gt_R.T @ gt_t).astype(np.float64)
            est_pts.append(est_center)
            gt_pts.append(gt_center)
        if not est_pts:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        return np.asarray(est_pts, dtype=np.float64), np.asarray(gt_pts, dtype=np.float64)

    def _align_est_to_gt(self, est_pts: np.ndarray, gt_pts: np.ndarray):
        """
        Align the estimated trajectory to GT using the evo-based flow from
        eval_utils; returns the original trajectory if there are too few
        points or evo is unavailable.
        """
        if est_pts.shape[0] < 3 or gt_pts.shape[0] < 3:
            return est_pts
        try:
            from evo.core import trajectory as evo_trajectory
            from evo.core.trajectory import PosePath3D
            from utils.eval_utils import detect_and_fix_axis_reflection

            def _pts_to_posepath(points_xyz: np.ndarray):
                poses = []
                for p in points_xyz:
                    T = np.eye(4, dtype=np.float64)
                    T[:3, 3] = p
                    poses.append(T)
                return PosePath3D(poses_se3=poses)

            traj_ref = _pts_to_posepath(gt_pts)
            traj_est = _pts_to_posepath(est_pts)
            traj_est_corrected, _ = detect_and_fix_axis_reflection(
                traj_est, traj_ref, correct_scale=True
            )
            traj_est_aligned = evo_trajectory.align_trajectory(
                traj_est_corrected, traj_ref, correct_scale=True
            )
            return traj_est_aligned.positions_xyz.astype(np.float64)
        except Exception:
            return est_pts

    def save_fixed_pinhole_render_with_trajectory(self, cur_frame_idx):
        if not self.enable_fixed_pinhole_render or not HAS_VISUALIZATION_DEPS:
            return
        if self.gaussians is None:
            return
        if cur_frame_idx % self.fixed_pinhole_render_interval != 0:
            return
        try:
            fixed_cam = self._build_fixed_pinhole_camera()
            with torch.no_grad():
                means3D = self.gaussians.get_xyz.to(dtype=torch.float32)
                if means3D.shape[0] == 0:
                    return
                means2D = torch.zeros_like(means3D, requires_grad=False)
                opacities = self.gaussians.get_opacity.unsqueeze(-1).to(dtype=torch.float32)
                scales = self.gaussians.get_scaling.to(dtype=torch.float32)
                rotations = self.gaussians.get_rotation.to(dtype=torch.float32)
                shs = self.gaussians.get_features.to(dtype=torch.float32)

                raster_settings = GaussianRasterizationSettings(
                    image_height=int(fixed_cam["image_height"]),
                    image_width=int(fixed_cam["image_width"]),
                    tanfovx=float(fixed_cam["tanfovx"]),
                    tanfovy=float(fixed_cam["tanfovy"]),
                    bg=self.background.to(dtype=torch.float32),
                    scale_modifier=1.0,
                    viewmatrix=fixed_cam["viewmatrix"],
                    projmatrix=fixed_cam["projmatrix"],
                    projmatrix_raw=fixed_cam["projmatrix_raw"],
                    sh_degree=int(self.gaussians.active_sh_degree),
                    campos=fixed_cam["campos"],
                    prefiltered=False,
                    debug=False,
                    rel_R=torch.eye(3, dtype=torch.float32, device=self.device),
                )
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                rendered = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    opacities=opacities,
                    shs=shs,
                    scales=scales,
                    rotations=rotations,
                )[0]
                img = rendered.detach().clamp(0.0, 1.0).cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img_uint8 = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            est_pts_w, gt_pts_w = self._collect_est_gt_trajectory_until(cur_frame_idx)
            est_pts_w_aligned = self._align_est_to_gt(est_pts_w, gt_pts_w)
            est_uv = self._project_world_points_to_fixed_camera(est_pts_w_aligned)
            gt_uv = self._project_world_points_to_fixed_camera(gt_pts_w)

            if self.fixed_pinhole_draw_gt and gt_uv.shape[0] >= 2:
                cv2.polylines(img_bgr, [gt_uv.reshape(-1, 1, 2)], False, (0, 0, 0), 2, lineType=cv2.LINE_AA)
                cv2.circle(img_bgr, tuple(gt_uv[-1]), 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            if self.fixed_pinhole_draw_est and est_uv.shape[0] >= 2:
                cv2.polylines(img_bgr, [est_uv.reshape(-1, 1, 2)], False, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                cv2.circle(img_bgr, tuple(est_uv[-1]), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            cv2.putText(
                img_bgr,
                f"frame={cur_frame_idx}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            if self.fixed_pinhole_draw_est:
                cv2.putText(
                    img_bgr, "EST", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA
                )
            if self.fixed_pinhole_draw_gt:
                cv2.putText(
                    img_bgr, "GT", (72, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, lineType=cv2.LINE_AA
                )

            save_path = os.path.join(self.fixed_pinhole_render_dir, f"fixed_pinhole_{cur_frame_idx:06d}.png")
            cv2.imwrite(save_path, img_bgr)
        except Exception as e:
            Log(f"Failed to save fixed pinhole render at frame {cur_frame_idx}: {e}")
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        valid_rgb = valid_rgb.squeeze(0)
        median_depth = 0.0
        # print("valid_rgb", valid_rgb.shape)
        if self.monocular:
            if depth is None:
                # Depth network or random initialization
                estimated_depth = None
                if self.use_depth_estimation and self.depth_estimator is not None:
                    try:
                        # Use the BiFuse depth estimator
                        estimated_depth = self.depth_estimator.estimate_depth(
                            gt_img, 
                            target_size=(gt_img.shape[1], gt_img.shape[2])
                        )
                        
                        # Depth scaling (scene dependent)
                        depth_scale = self.config.get("depth_estimation", {}).get("depth_scale", 1.0)
                        initial_depth = estimated_depth.unsqueeze(0) * depth_scale
                        initial_depth += torch.randn_like(initial_depth) * 0.2
                        model_type = self.config.get('depth_estimation', {}).get('model', 'bifuse')
                        Log(f"BiFuse depth estimation, frame {cur_frame_idx}")
                    except Exception as e:
                        Log(f"Depth estimation failed: {e}, using random depth")
                        estimated_depth = None
                        initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                        initial_depth += torch.randn_like(initial_depth) * 0.3
                else:
                    # Improved random initialization for panoramic cameras
                    sensor_type = self.config.get("Dataset", {}).get("sensor_type", "").lower()
                    # if sensor_type == "panoramic":
                    #     # smoother depth init for panoramic cameras
                    #     initial_depth = self._create_panoramic_depth_init(gt_img)
                    # else:
                        # Original random initialization
                    initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                    initial_depth += torch.randn_like(initial_depth) * 0.3
                
                # Save depth-estimation visualization
                if self.save_depth_visualization and (init or cur_frame_idx % 10 == 0):  # at init and every 10 frames
                    self.save_depth_visualization_data(cur_frame_idx, gt_img, estimated_depth, initial_depth, valid_rgb)
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
               
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:

                    # depth_filled, local_std, valid_mask = get_median_depth(
                    #     depth, opacity, mask=valid_rgb, return_std=True
                    # )
                    # invalid_depth_mask = ~valid_mask
                    # initial_depth = depth_filled + torch.randn_like(depth_filled) * torch.where(
                    #     invalid_depth_mask, local_std * 0.5, local_std * 0.2
                    # )
                    
                    depth_median, local_std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    depth_2d = depth.squeeze(0)
                    valid_mask_2d = valid_mask.squeeze(0) if valid_mask.ndim == 3 else valid_mask
                    depth_filled = self._fill_invalid_depth_with_nearest(
                        depth_2d, valid_mask_2d, depth_median
                    ).unsqueeze(0)
                    
                    invalid_depth_mask = ~valid_mask
                    initial_depth = depth_filled + torch.randn_like(depth_filled) * torch.where(
                        invalid_depth_mask, local_std * 0.5, local_std * 0.2
                    )
            if initial_depth.ndimension() == 3:
                return initial_depth.cpu().numpy()[0]  # first channel if 3D
            else:
                return initial_depth.cpu().numpy()  # already 2D
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def rotation_matrix_to_axis_angle(self, R):
        """
        Convert a rotation matrix to axis-angle (lie algebra).
        R: 3x3 rotation matrix
        Returns: 3-vector (axis-angle)
        """
        # Inverse Rodrigues formula
        theta = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1.0, 1.0))
        
        if theta.abs() < 1e-6:
            # Tiny rotation; return zero
            return torch.zeros(3, device=R.device)
        
        # Extract the rotation axis
        w = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * torch.sin(theta))
        
        return w * theta
    
    def axis_angle_to_rotation_matrix(self, w):
        """
        Convert axis-angle (lie algebra) to a rotation matrix.
        w: 3-vector (axis-angle)
        Returns: 3x3 rotation matrix
        """
        theta = torch.norm(w)
        
        if theta < 1e-6:
            # Tiny rotation; return identity
            return torch.eye(3, device=w.device)
        
        # Normalize the rotation axis
        k = w / theta
        
        # Build the skew-symmetric matrix
        K = torch.tensor([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], device=w.device)
        
        # Rodrigues formula
        R = torch.eye(3, device=w.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        return R

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        
        # Reset the velocity estimate
        self.velocity_T = None
        self.velocity_R = None
        
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)


        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        start_time = time.time()
        
        # Profiling: pose initialization
        self.profiler.start("tracking_pose_init")
        prev_idx = cur_frame_idx - self.use_every_n_frames
        prev = self.cameras[prev_idx]
        viewpoint.update_RT(prev.R, prev.T)

        self.profiler.end()
        
        # Profiling: optimizer setup
        self.profiler.start("tracking_optimizer_setup")
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        
        pose_optimizer = torch.optim.Adam(opt_params)
        self.profiler.end()

        
        for tracking_itr in range(self.tracking_itr_num):
            
            # Profiling: rendering
            self.profiler.start("tracking_render")
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            
            self.profiler.end()
            
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            # Clear the previous iteration gradients
            pose_optimizer.zero_grad()
            # Profiling: loss computation
            self.profiler.start("tracking_loss")
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            self.profiler.end()
            
            # Profiling: backward
            self.profiler.start("tracking_backward")
            loss_tracking.backward()
            self.profiler.end()
            
            with torch.no_grad():
                self.last_render_image = render_pkg["render"]
                self.last_render_opacity = render_pkg["opacity"]  # save opacity for next-frame sampling
        
                 # Save rendered images
                if self.save_render_images and cur_frame_idx % self.render_images_interval == 0 and tracking_itr == 0:
                    # Save the rendered image
                    image_np = image.detach().cpu().numpy()
                    image_np = np.transpose(image_np, (1, 2, 0))
                    image_uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
                    render_image_path = os.path.join(self.render_images_dir, f"render_frame_{cur_frame_idx:06d}.png")
                    Image.fromarray(image_uint8).save(render_image_path)
                
                # Save rendered depth at the same interval (raw + colormap)
                if self.save_render_depth and cur_frame_idx % self.render_images_interval == 0 and tracking_itr == 0:
                    depth_np = depth.detach().cpu().numpy()
                    if depth_np.ndim == 3:
                        depth_np = depth_np[0] if depth_np.shape[0] == 1 else np.squeeze(depth_np)
                    depth_np = depth_np.astype(np.float32)
                    
                    depth_npy_path = os.path.join(self.render_depth_dir, f"render_depth_{cur_frame_idx:06d}.npy")
                    np.save(depth_npy_path, depth_np)
                    
                    valid_mask = np.isfinite(depth_np) & (depth_np > 0)
                    if np.any(valid_mask):
                        depth_min = float(depth_np[valid_mask].min())
                        depth_max = float(depth_np[valid_mask].max())
                        depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
                        if depth_max > depth_min:
                            depth_norm = (depth_np[valid_mask] - depth_min) / (depth_max - depth_min + 1e-8)
                            depth_vis[valid_mask] = (depth_norm * 255).astype(np.uint8)
                        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        depth_png_path = os.path.join(self.render_depth_dir, f"render_depth_{cur_frame_idx:06d}.png")
                        Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)).save(depth_png_path)
                
                # Profiling: optimizer and pose update
                # Note: torch.cuda.Event would time more precisely than time.time()
                self.profiler.start("tracking_optimizer_step")
                pose_optimizer.step()
                self.profiler.end()
                
                self.profiler.start("tracking_update_pose")
                converged = update_pose(viewpoint)
                self.profiler.end()


            if tracking_itr % 2 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
        total_tracking_time = time.time() - start_time
        self.profiler.record("tracking_total_time", total_tracking_time)
        # print(f"Tracking time: {total_tracking_time:.3f} seconds")
        
        # Profiling: median depth
        self.profiler.start("tracking_median_depth")
        self.median_depth = get_median_depth(depth, opacity)
        self.profiler.end()
        
        

        # Optional: fixed-viewpoint pinhole render with trajectories
        self.save_fixed_pinhole_render_with_trajectory(cur_frame_idx)
        
        return render_pkg

    def save_pose_as_json(self, filepath, cam_dict):
        extrinsics = []
        for key, vp in sorted(cam_dict.items()):
            R = vp.R.detach().cpu().numpy().tolist()
            T = vp.T.detach().cpu().numpy()
            # Camera center C = -R^T @ T
            R_np = np.array(R)
            center = (-R_np.T @ T).tolist()

            extrinsics.append({
                "key": key,
                "value": {
                    "rotation": R,
                    "center": center
                }
            })

        with open(filepath, "w") as f:
            json.dump({"extrinsics": extrinsics}, f, indent=4)

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        # pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        pose_W2C = getWorld2View2(curr_frame.R, curr_frame.T)
        # last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_W2C = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_C2W = torch.linalg.inv(last_kf_W2C)
        dist = torch.norm((pose_W2C @ last_kf_C2W)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None

        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            denom = (cur_frame_visibility_filter.count_nonzero()+occ_aware_visibility[kf_idx].count_nonzero())/2
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4

            # cut_off = 0.1
            # print(cut_off)
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))


        if len(window) > self.config["Training"]["window_size"]:
            # Precompute keyframe transforms to avoid recomputation
            # World-to-camera transforms of all keyframes
            kf_WC_list = []
            kf_CW_list = []
            for idx in range(N_dont_touch, len(window)):
                kf_idx = window[idx]
                kf = self.cameras[kf_idx]
                kf_CW = getWorld2View2(kf.R, kf.T)
                kf_WC = torch.linalg.inv(kf_CW)
                kf_CW_list.append(kf_CW)
                kf_WC_list.append(kf_WC)
            
            # Distance from the current frame to all keyframes (for k)
            T_C0_list = []
            for kf_CW in kf_CW_list:
                T_CiC0 = kf_CW @ kf_0_WC
                T_C0_list.append(T_CiC0)
            
            # Pairwise keyframe distances (computed once)
            # Symmetry: dist(i,j) = dist(j,i)
            n_kfs = len(kf_CW_list)
            dist_matrix = torch.zeros((n_kfs, n_kfs), device=kf_0_WC.device)
            for i in range(n_kfs):
                for j in range(i + 1, n_kfs):
                    T_CiCj = kf_CW_list[i] @ kf_WC_list[j]
                    dist = torch.norm(T_CiCj[0:3, 3])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist  # symmetric
            
            # Set the diagonal to a large value to minimize its effect
            # Use fill_diagonal_ (matrix must be non-empty)
            if n_kfs > 0:
                # Mask to exclude the diagonal
                mask = ~torch.eye(n_kfs, dtype=torch.bool, device=dist_matrix.device)
                
                # inv_dist per keyframe (vectorized)
                inv_dist = []
                for i in range(n_kfs):
                    # Sum of inverse distances to other keyframes (diagonal masked)
                    row_dist = dist_matrix[i, :]
                    inv_dists = 1.0 / (row_dist + 1e-6)
                    inv_dists_sum = inv_dists[mask[i]].sum().item()
                    
                    # Distance k to the current frame
                    k = torch.sqrt(torch.norm(T_C0_list[i][0:3, 3])).item()
                    inv_dist.append(k * inv_dists_sum)
            else:
                inv_dist = []

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            # print("removed_frame: ", removed_frame)
            window.remove(removed_frame)
        
        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        # Detach opacity so it can cross process boundaries
        last_render_opacity_detached = None
        if self.last_render_opacity is not None:
            last_render_opacity_detached = self.last_render_opacity.detach()
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, last_render_opacity_detached]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())


    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        # Run from the configured start frame
        cur_frame_idx = self.start_frame
        Log(f"Starting SLAM from frame {cur_frame_idx}")
        
        # projection_matrix = get_spherical_projection_matrix(width = self.dataset.width, height = self.dataset.height)
        # The projection matrix is computed once and already on the right device
        # Use the cached projection matrix
        if not hasattr(self, '_cached_projection_matrix'):
            self._cached_projection_matrix = getProjectionMatrix2(
                znear=0.01,
                zfar=100.0,
                fx=self.dataset.fx,
                fy=self.dataset.fy,
                cx=self.dataset.cx,
                cy=self.dataset.cy,
                W=self.dataset.width,
                H=self.dataset.height,
            ).transpose(0, 1).to(device=self.device)
        projection_matrix = self._cached_projection_matrix
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                start_time = time.time()
                if cur_frame_idx >= len(self.dataset.image_paths):
                    if self.save_results:
                        # Evaluate ATE
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        
                        # Save the full TUM-format trajectory
                        Log("Saving full-frame TUM trajectory")
                        from utils.eval_utils import save_trajectory_tum_format, save_trajectory_tum_format_keyframes_only
                        save_trajectory_tum_format(
                            self.cameras, 
                            self.save_dir, 
                            filename="trajectory_all_frames.txt"
                        )
                        
                        # Save the keyframe TUM trajectory
                        Log("Saving keyframe TUM trajectory")
                        save_trajectory_tum_format_keyframes_only(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            filename="trajectory_keyframes.txt"
                        )
                        
                        # Save the final optimized point cloud
                        Log("Saving final point cloud map")
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                # if self.requested_keyframe > 0:
                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                print("cur_frame_idx : ",cur_frame_idx)
                # if self.gaussians is not None:
                #     print(self.gaussians._xyz.shape[0])
                
                # Profiling: camera initialization
                self.profiler.start("camera_init")
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                self.profiler.end()

                # Profiling: gradient mask
                self.profiler.start("compute_grad_mask")
                viewpoint.compute_grad_mask(self.config)
                self.profiler.end()

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )
                # Tracking
                self.profiler.start("tracking_total")
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                self.profiler.end()
# Profiling: prepare GUI data
                self.profiler.start("prepare_gui_data")
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
                self.profiler.end()
                
                # Profiling: GUI communication
                self.profiler.start("gui_communication")
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )
                self.profiler.end()

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1

                    continue

                # Profiling: keyframe decision
                self.profiler.start("is_keyframe_check")
                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval

                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                self.profiler.end()
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    # print("point_ratio: ", point_ratio)
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"] )

                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    # Profiling: add to window
                    self.profiler.start("add_to_window")
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    Log("key_frame_idx: ", cur_frame_idx)
                    Log("current_window: ", self.current_window)
                    self.profiler.end()
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    

                    # Profiling: add keyframe
                    self.profiler.start("add_new_keyframe")
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.profiler.end()
                    
                    # Profiling: request keyframe (backend)
                    self.profiler.start("request_keyframe")
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                    self.profiler.end()

                else:
                    # Profiling: cleanup
                    self.profiler.start("cleanup")
                    self.cleanup(cur_frame_idx)
                    self.profiler.end()
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
                
                # Profiling: print stats
                total_frame_time = time.time() - start_time
                self.profiler.record("frame_total", total_frame_time)
                
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1
                    Log("backend finished keyframe")

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
