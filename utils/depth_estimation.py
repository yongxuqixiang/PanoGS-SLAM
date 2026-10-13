"""
Panoramic depth prior based on BiFuse (monocular 360 depth estimation).
Used to bootstrap the Gaussian map from the first frame.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Tuple, Union

# Add the BiFuse submodule to the Python path
BIFUSE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "BiFuse")
if BIFUSE_PATH not in sys.path:
    sys.path.insert(0, BIFUSE_PATH)


class BiFuseDepthEstimator:
    """
    BiFuse panoramic depth estimator for 360-degree equirectangular images.
    """
    
    def __init__(
        self, 
        pretrained_path: str = "BiFuse_Pretrained.pkl",
        device: str = "cuda",
        target_height: int = 512,
        target_width: int = 1024
    ):
        """
        Initialize the BiFuse depth estimator.

        Args:
            pretrained_path: path to the pretrained weights
            device: compute device ('cuda' or 'cpu')
            target_height: input height expected by BiFuse (512)
            target_width: input width expected by BiFuse (1024)
        """
        self.device = device
        self.target_height = target_height
        self.target_width = target_width
        self.pretrained_path = pretrained_path
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained BiFuse model."""
        try:
            # Import the BiFuse model
            from models.FCRN import MyModel as BiFuseModel
            
            # Create the model instance
            self.model = BiFuseModel(
                layers=50,
                decoder="upproj",
                output_size=None,
                in_channels=3,
                pretrained=True
            )
            
            # Load pretrained weights
            if os.path.exists(self.pretrained_path):
                print(f"Loading BiFuse pretrained weights: {self.pretrained_path}")
                params = torch.load(self.pretrained_path, map_location='cpu')
                self.model.load_state_dict(params, strict=False)
            else:
                print(f"Warning: pretrained weights not found: {self.pretrained_path}")
                print("Falling back to randomly initialized weights")
            
            # Move to the target device and switch to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("BiFuse model loaded")
            
        except Exception as e:
            print(f"Error: failed to load BiFuse model: {e}")
            print("Falling back to random depth")
            self.model = None
    
    def estimate_depth(
        self, 
        image: torch.Tensor, 
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Estimate the depth of a panoramic image.

        Args:
            image: input panorama [C, H, W] or [H, W, C], values in [0, 1]
            target_size: desired output size (height, width); uses the input size if None

        Returns:
            depth: depth map [H, W] with metric values (meters)
        """
        if self.model is None:
            return self._random_depth_fallback(image, target_size)
        
        try:
            # Preprocess the image
            input_tensor = self._preprocess_image(image)
            
            # Run depth estimation
            with torch.no_grad():
                # BiFuse returns three outputs: (equi_depth, cube_depth, refine_depth)
                # We use the final refine_depth
                _, _, depth_pred = self.model(input_tensor)
            
            # Postprocess the depth map
            depth = self._postprocess_depth(depth_pred, image, target_size)
            
            return depth
            
        except Exception as e:
            print(f"Warning: BiFuse depth estimation failed: {e}")
            print("Falling back to random depth")
            return self._random_depth_fallback(image, target_size)
    
    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input image for BiFuse.

        Args:
            image: input image [C, H, W] or [H, W, C]

        Returns:
            preprocessed: preprocessed image [1, C, H, W]
        """
        # Ensure a 3D tensor
        if image.dim() == 4:
            image = image.squeeze(0)
        
        # Convert to [C, H, W]
        if image.shape[0] != 3:
            # Assume [H, W, C] layout
            image = image.permute(2, 0, 1)
        
        # Clamp to [0, 1]
        image = torch.clamp(image, 0, 1)
        
        # Resize via numpy (matching the BiFuse training preprocessing)
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        
        # Resize to the canonical BiFuse size (512, 1024)
        image_resized = cv2.resize(
            image_np, 
            (self.target_width, self.target_height), 
            interpolation=cv2.INTER_AREA
        )
        
        # Back to tensor
        image_tensor = torch.from_numpy(image_resized).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # [C, H, W]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def _postprocess_depth(
        self, 
        depth_pred: torch.Tensor, 
        original_image: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Postprocess the raw depth prediction.

        Args:
            depth_pred: predicted depth [1, 1, H, W]
            original_image: original input image (used to infer the output size)
            target_size: desired output size

        Returns:
            depth: postprocessed depth map [H, W]
        """
        # Remove batch and channel dimensions
        depth = depth_pred.squeeze().cpu()
        
        # BiFuse outputs depths in [0, 10] m; clamp to this range
        depth = torch.clamp(depth, 0, 10)
        
        # Determine the target size
        if target_size is None:
            # Fall back to the original image size
            if original_image.dim() == 4:
                original_image = original_image.squeeze(0)
            
            if original_image.shape[0] == 3:  # [C, H, W]
                target_size = (original_image.shape[1], original_image.shape[2])
            else:  # [H, W, C]
                target_size = (original_image.shape[0], original_image.shape[1])
        
        # Resize to the target size
        if depth.shape != target_size:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        return depth
    
    def _random_depth_fallback(
        self, 
        image: torch.Tensor, 
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Random depth fallback (used when the network is unavailable).

        Args:
            image: input image
            target_size: desired output size

        Returns:
            depth: random depth map [H, W]
        """
        if target_size is None:
            if image.dim() == 4:
                image = image.squeeze(0)
            
            if image.dim() == 3 and image.shape[0] == 3:
                target_size = (image.shape[1], image.shape[2])
            else:
                target_size = (image.shape[0], image.shape[1])
        
        # Random depth around 2 m (std 0.3 m)
        depth = 2.0 * torch.ones(target_size)
        depth += torch.randn_like(depth) * 0.3
        depth = torch.clamp(depth, 0.5, 10.0)  # keep within a reasonable range
        
        return depth
    
    def estimate_depth_batch(
        self, 
        images: torch.Tensor, 
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Estimate depths for a batch of images.

        Args:
            images: input images [B, C, H, W]
            target_size: desired output size

        Returns:
            depths: depth maps [B, H, W]
        """
        depths = []
        for i in range(images.shape[0]):
            depth = self.estimate_depth(images[i], target_size)
            depths.append(depth)
        
        return torch.stack(depths)


def create_depth_estimator(config: dict) -> BiFuseDepthEstimator:
    """
    Create the panoramic depth estimator from the experiment config.

    Args:
        config: full experiment config dict

    Returns:
        A BiFuseDepthEstimator instance
    """
    depth_config = config.get("depth_estimation", {})
    device = config.get("model_params", {}).get("data_device", "cuda")

    pretrained_path = depth_config.get("pretrained_path", "BiFuse_Pretrained.pkl")
    target_height = depth_config.get("target_height", 512)
    target_width = depth_config.get("target_width", 1024)

    estimator = BiFuseDepthEstimator(
        pretrained_path=pretrained_path,
        device=device,
        target_height=target_height,
        target_width=target_width
    )
    print(f"BiFuse depth estimator created (target_size={target_height}x{target_width})")
    return estimator
