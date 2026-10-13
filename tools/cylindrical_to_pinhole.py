#!/usr/bin/env python3
"""
Batch-convert equirectangular (EQR, 2:1) panoramas to pinhole images
Batch-convert equirectangular (EQR, 2:1) panoramas to pinhole images
Fast remap-based conversion, with camera intrinsics/extrinsics output
"""

import numpy as np
import cv2
import os
import json
import argparse
from pathlib import Path
from typing import List
from typing import List


class EquirectangularToPinhole:
    """Equirectangular (EQR) panorama to pinhole image converter."""
    def __init__(
        self,
        output_width: int = 640,
        output_height: int = 480,
        fov: float = 90.0,
        vfov: float = 60.0,
    ):
        """
        Initialize the converter.
        
        Args:
            output_width: output image width
            output_height: output image height
            fov: horizontal FoV (deg), default 90
            vfov: vertical FoV (deg), default 60
            Note: only equirectangular input is supported (aspect ratio ~ 2:1)
            fov: horizontal FoV (deg), default 90
            vfov: vertical FoV (deg), default 60
            Note: only equirectangular input is supported (aspect ratio ~ 2:1)
        """
        self.output_width = output_width
        self.output_height = output_height
        self.fov = fov
        self.vfov = vfov
        self.vfov = vfov
        
        # Compute the camera intrinsics
        self.fx = self.output_width / (2.0 * np.tan(np.radians(self.fov / 2.0)))
        self.fy = self.output_height / (2.0 * np.tan(np.radians(self.vfov / 2.0)))
        self.fx = self.output_width / (2.0 * np.tan(np.radians(self.fov / 2.0)))
        self.fy = self.output_height / (2.0 * np.tan(np.radians(self.vfov / 2.0)))
        self.cx = self.output_width / 2.0
        self.cy = self.output_height / 2.0
        
        # Precompute the mapping for speed
        self.map_x = None
        self.map_y = None
        self.pano_width = None
        self.pano_height = None
        self._last_yaw_offset = None
        self._last_pitch_offset = None
        self.pano_width = None
        self.pano_height = None
        self._last_yaw_offset = None
        self._last_pitch_offset = None
        
    def get_intrinsics(self) -> np.ndarray:
        """Return the camera intrinsic matrix (3x3)."""
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        return K
    
    def get_extrinsics(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                      translation: np.ndarray = None) -> np.ndarray:
        """
        Build the camera extrinsic matrix (4x4).
        
        Args:
            yaw: yaw angle (deg), about Y
            pitch: pitch angle (deg), about X
            roll: roll angle (deg), about Z
            translation: translation vector (x, y, z)
            
        Returns:
            4x4 extrinsic matrix [R|t]
        """
        # Convert to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Build the rotation matrices
        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        Rz = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations (Rz * Ry * Rx)
        R = Rz @ Ry @ Rx
        
        # Default translation: origin
        if translation is None:
            translation = np.array([0.0, 0.0, 0.0])
        
        # Build the 4x4 extrinsic matrix
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = translation
        
        return extrinsics
     
    def _build_mapping(self, pano_width: int, pano_height: int, 
                      yaw_offset: float = 0.0, pitch_offset: float = 0.0):
        """
        Precompute the EQR panorama -> pinhole mapping.
        Precompute the EQR panorama -> pinhole mapping.
        
        Args:
            pano_width: panorama width
            pano_height: panorama height
            pano_width: panorama width
            pano_height: panorama height
            yaw_offset: yaw offset (deg)
            pitch_offset: pitch offset (deg)
        """
        # Reuse the cached map when size/viewpoint are unchanged
        if (
            self.map_x is not None
            and self.pano_width == pano_width
            and self.pano_height == pano_height
            and self._last_yaw_offset == float(yaw_offset)
            and self._last_pitch_offset == float(pitch_offset)
        ):
            return
        
        self.pano_width = pano_width
        self.pano_height = pano_height
        self._last_yaw_offset = float(yaw_offset)
        self._last_pitch_offset = float(pitch_offset)
        self.pano_width = pano_width
        self.pano_height = pano_height
        self._last_yaw_offset = float(yaw_offset)
        self._last_pitch_offset = float(pitch_offset)
        
        # Pixel coordinate grid of the output image
        u, v = np.meshgrid(np.arange(self.output_width), np.arange(self.output_height))
        
        # Normalize coordinates
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = np.ones_like(x)
        
        # 3D ray directions
        norm = np.sqrt(x**2 + y**2 + z**2)
        x /= norm
        y /= norm
        z /= norm
        
        # Apply the viewpoint rotation
        yaw_rad = np.radians(yaw_offset)
        pitch_rad = np.radians(pitch_offset)
        
        # Pitch rotation (about X)
        if pitch_offset != 0:
            y_rot = y * np.cos(pitch_rad) - z * np.sin(pitch_rad)
            z_rot = y * np.sin(pitch_rad) + z * np.cos(pitch_rad)
            y = y_rot
            z = z_rot
        
        # Yaw rotation (about Y)
        if yaw_offset != 0:
            x_rot = x * np.cos(yaw_rad) + z * np.sin(yaw_rad)
            z_rot = -x * np.sin(yaw_rad) + z * np.cos(yaw_rad)
            x = x_rot
            z = z_rot
        
        # EQR: u <-> horizontal angle theta; v <-> pitch phi ([-pi/2, pi/2])
        # EQR: u <-> horizontal angle theta; v <-> pitch phi ([-pi/2, pi/2])
        theta = np.arctan2(x, z)
        # Note: image v points down, so the normalized y also points down, while
        # the EQR pitch phi is defined positive-up; we use -y to avoid flipping.
        phi = np.arctan2(-y, np.sqrt(x**2 + z**2))
        self.map_x = ((theta + np.pi) / (2.0 * np.pi) * pano_width).astype(np.float32)
        self.map_y = (((np.pi / 2.0) - phi) / np.pi * pano_height).astype(np.float32)
        # Note: image v points down, so the normalized y also points down, while
        # the EQR pitch phi is defined positive-up; we use -y to avoid flipping.
        phi = np.arctan2(-y, np.sqrt(x**2 + z**2))
        self.map_x = ((theta + np.pi) / (2.0 * np.pi) * pano_width).astype(np.float32)
        self.map_y = (((np.pi / 2.0) - phi) / np.pi * pano_height).astype(np.float32)
        
        # Wrap horizontally (seamless), clip vertically to the valid range
        self.map_x = np.mod(self.map_x, pano_width).astype(np.float32)
        self.map_y = np.clip(self.map_y, 0, pano_height - 1).astype(np.float32)
        # Wrap horizontally (seamless), clip vertically to the valid range
        self.map_x = np.mod(self.map_x, pano_width).astype(np.float32)
        self.map_y = np.clip(self.map_y, 0, pano_height - 1).astype(np.float32)
    
    def convert(self, cylindrical_img: np.ndarray, 
                yaw_offset: float = 0.0, 
                pitch_offset: float = 0.0) -> np.ndarray:
        """
        Convert an EQR panorama to a pinhole image.
        Convert an EQR panorama to a pinhole image.
        
        Args:
            cylindrical_img: input EQR panorama
            cylindrical_img: input EQR panorama
            yaw_offset: horizontal view offset (deg), 0 = forward
            pitch_offset: vertical view offset (deg), 0 = level
            
        Returns:
            the converted pinhole image
        """
        h, w = cylindrical_img.shape[:2]
        
        # Build or update the mapping
        self._build_mapping(w, h, yaw_offset, pitch_offset)
        
        # Fast remapping via OpenCV remap
        # Note: map_x is already wrapped (mod), so no BORDER_WRAP is needed here;
        # Avoid vertical wrapping, which would create fake seams at the top/bottom.
        pinhole_img = cv2.remap(
            cylindrical_img,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Note: map_x is already wrapped (mod), so no BORDER_WRAP is needed here;
        # Avoid vertical wrapping, which would create fake seams at the top/bottom.
        pinhole_img = cv2.remap(
            cylindrical_img,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
        return pinhole_img
    
    def save_camera_params(self, output_path: str, yaw: float = 0.0, 
                          pitch: float = 0.0, roll: float = 0.0,
                          translation: np.ndarray = None):
        """
        Save the camera intrinsics/extrinsics to a JSON file.
        
        Args:
            output_path: output file path
            yaw, pitch, roll: rotation angles (deg)
            translation: translation vector
        """
        K = self.get_intrinsics()
        extrinsics = self.get_extrinsics(yaw, pitch, roll, translation)
        
        params = {
            "intrinsics": {
                "K": K.tolist(),
                "fx": float(self.fx),
                "fy": float(self.fy),
                "cx": float(self.cx),
                "cy": float(self.cy),
                "width": self.output_width,
                "height": self.output_height,
                "hfov_degrees": float(self.fov),
                "vfov_degrees": float(self.vfov)
            },
            "extrinsics": {
                "matrix": extrinsics.tolist(),
                "rotation": extrinsics[:3, :3].tolist(),
                "translation": extrinsics[:3, 3].tolist(),
                "yaw_degrees": float(yaw),
                "pitch_degrees": float(pitch),
                "roll_degrees": float(roll)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)


def batch_convert(input_dir: str, output_dir: str, 
                 width: int = 640, height: int = 480, fov: float = 90.0, vfov: float = 60.0,
                 yaw_angles: List[float] = None, pitch_angle: float = 0.0,
                 extensions: List[str] = None):
    """
    Batch-convert all panoramas in a directory.
    
    Args:
        input_dir: input directory
        output_dir: output directory
        width: output image width
        height: output image height
        fov: field of view (deg)
        yaw_angles: yaw angles (deg) for multi-view output; default [0]
        pitch_angle: pitch angle (deg)
        extensions: supported image extensions
    """
    if yaw_angles is None:
        yaw_angles = [0.0]
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Create the output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create the converter (EQR input only)
    converter = EquirectangularToPinhole(width, height, fov, vfov=vfov)
    # Create the converter (EQR input only)
    converter = EquirectangularToPinhole(width, height, fov, vfov=vfov)
    
    # Collect the image files
    input_path = Path(input_dir)
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    
    image_files = sorted(set(image_files))
    
    if len(image_files) == 0:
        print(f"warning: no images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Output size: {width}x{height}, FOV: {fov} deg")
    print(f"Generating {len(yaw_angles)} views: {yaw_angles}")
    
    # Save the camera parameters
    for i, yaw in enumerate(yaw_angles):
        param_file = output_path / f"camera_params_yaw{int(yaw)}.json"
        converter.save_camera_params(str(param_file), yaw=yaw, pitch=pitch_angle)
        print(f"Camera parameters saved: {param_file}")
    
    # Batch process the images
    for idx, img_file in enumerate(image_files):
        print(f"Processing [{idx+1}/{len(image_files)}]: {img_file.name}")
        
        # Read the image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  warning: cannot read {img_file}")
            continue
        
        # Generate one image per yaw angle
        for yaw in yaw_angles:
            # Convert the image
            pinhole_img = converter.convert(img, yaw_offset=yaw, pitch_offset=pitch_angle)
            
            # Build the output filename
            stem = img_file.stem
            if len(yaw_angles) > 1:
                output_name = f"{stem}_yaw{int(yaw)}{img_file.suffix}"
            else:
                output_name = f"{stem}{img_file.suffix}"
                output_name = f"{stem}{img_file.suffix}"
            
            output_file = output_path / output_name
            
            # Save the image
            cv2.imwrite(str(output_file), pinhole_img)
        
        if len(yaw_angles) == 1:
            print(f"  -> {output_name}")
        else:
            print(f"  -> generated {len(yaw_angles)} views")
    
    print(f"\nDone! Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch-convert equirectangular (EQR) panoramas to pinhole images'
    )
    parser.add_argument(
    '--input_dir',
    type=str,
    default='test_demo/test',
    help='input image directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_demo/pinhole',
        help='output image directory'
    )
    parser.add_argument('--width', type=int, default=960, help='output image width')
    parser.add_argument('--height', type=int, default=480, help='output image height')
    parser.add_argument('--fov', type=float, default=120.0, help='horizontal FoV (deg)')
    parser.add_argument('--vfov', type=float, default=60.0, help='vertical FoV (deg)')
    parser.add_argument('--yaw', type=str, default='0', 
                       help='yaw angles (deg), comma-separated, e.g. "0,90,180,270"')
    parser.add_argument('--pitch', type=float, default=0.0, help='pitch angle (deg)')
    
    args = parser.parse_args()
    
    # Parse the yaw list
    yaw_angles = [float(y.strip()) for y in args.yaw.split(',')]
    
    # Run the batch conversion
    batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        fov=args.fov,
        vfov=args.vfov,
        yaw_angles=yaw_angles,
        pitch_angle=args.pitch,
    )


if __name__ == '__main__':
    main()

# Backward-compatible alias; the implementation supports EQR input only.
CylindricalToPinhole = EquirectangularToPinhole
# Backward-compatible alias; the implementation supports EQR input only.
CylindricalToPinhole = EquirectangularToPinhole
