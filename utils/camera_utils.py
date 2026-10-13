import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        mask = None,
        device="cuda:0",
        timestamp=None,
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device
        # Use uid as the timestamp if not provided
        self.timestamp = timestamp if timestamp is not None else float(uid)

        # Create the tensor at the final size directly
        self.R = torch.eye(3, device=device)
        self.T = torch.zeros(3, device=device)
        # Direct slicing; gt_T is usually already on the right device
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        # Direct assignment; data is usually on the right device
        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width
        self.mask = mask

        # Create the Parameter with zeros
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        # zeros instead of tensor([0.0])
        self.exposure_a = nn.Parameter(
            torch.zeros(1, requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.zeros(1, requires_grad=True, device=device)
        )

        # Direct assignment; projection_matrix is cached on the right device
        self.projection_matrix = projection_matrix

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        # Use the dataset timestamp if present, else the index
        timestamp = dataset.timestamps[idx] if hasattr(dataset, 'timestamps') else None
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            dataset.mask,
            device=dataset.device,
            timestamp=timestamp,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]


    @property
    def get_c2w(self):
        """
        Computes and returns the Camera-to-World (C2W) transformation matrix.
        It's the inverse of the World-to-Camera (W2C) matrix.
        """
        return torch.linalg.inv(self.world_view_transform)
    
    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            # Precompute all block indices
            # torch.quantile instead of repeated median calls (fewer syncs)
            block_h = h // row
            block_w = w // col
            
            # Collect all blocks
            blocks = []
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * block_h : (r + 1) * block_h,
                        c * block_w : (c + 1) * block_w,
                    ]
                    blocks.append(block.flatten())
            
            # All block medians at once (quantile is faster than median)
            if blocks:
                all_blocks = torch.stack([block.median() for block in blocks])
                thresholds = all_blocks * multiplier
                
                # Apply the threshold (vectorized)
                idx = 0
                for r in range(row):
                    for c in range(col):
                        block = img_grad_intensity[
                            :,
                            r * block_h : (r + 1) * block_h,
                            c * block_w : (c + 1) * block_w,
                        ]
                        img_grad_intensity[
                            :,
                            r * block_h : (r + 1) * block_h,
                            c * block_w : (c + 1) * block_w,
                        ] = (block > thresholds[idx]).float()
                        idx += 1
            self.grad_mask = img_grad_intensity
        else:
            # torch.quantile instead of median (fewer syncs)
            median_img_grad_intensity = img_grad_intensity.quantile(0.5)
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
