import argparse
from pathlib import Path

import cv2
import numpy as np


def _clamp_ratio(value: float, name: str) -> float:
    if not 0.0 <= value <= 0.5:
        raise ValueError(f"{name} must be in [0.0, 0.5], got {value}")
    return value


def build_mask(
    width: int,
    height: int,
    top_ratio: float,
    bottom_ratio: float,
    left_ratio: float,
    right_ratio: float,
) -> np.ndarray:
    """
    Build a single-channel mask:
    - 255 (white) marks invalid regions
    - 0   (black) marks valid regions
    """
    top_ratio = _clamp_ratio(top_ratio, "top_ratio")
    bottom_ratio = _clamp_ratio(bottom_ratio, "bottom_ratio")
    left_ratio = _clamp_ratio(left_ratio, "left_ratio")
    right_ratio = _clamp_ratio(right_ratio, "right_ratio")

    mask = np.zeros((height, width), dtype=np.uint8)

    top_h = int(round(height * top_ratio))
    bottom_h = int(round(height * bottom_ratio))
    left_w = int(round(width * left_ratio))
    right_w = int(round(width * right_ratio))

    if top_h > 0:
        mask[:top_h, :] = 255
    if bottom_h > 0:
        mask[height - bottom_h :, :] = 255
    if left_w > 0:
        mask[:, :left_w] = 255
    if right_w > 0:
        mask[:, width - right_w :] = 255

    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a border invalid-region mask: white=invalid, black=valid"
    )
    parser.add_argument("--width", type=int, default=960, help="mask width")
    parser.add_argument("--height", type=int, default=480, help="mask height")
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.25,
        help="invalid fraction of the height at the top",
    )
    parser.add_argument(
        "--bottom-ratio",
        type=float,
        default=0.25,
        help="invalid fraction of the height at the bottom",
    )
    parser.add_argument(
        "--left-ratio",
        type=float,
        default=0.0,
        help="invalid fraction of the width on the left",
    )
    parser.add_argument(
        "--right-ratio",
        type=float,
        default=0.0,
        help="invalid fraction of the width on the right",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_demo/mask_1024x512.png",
        help="output image path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive integers")

    mask = build_mask(
        width=args.width,
        height=args.height,
        top_ratio=args.top_ratio,
        bottom_ratio=args.bottom_ratio,
        left_ratio=args.left_ratio,
        right_ratio=args.right_ratio,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), mask)
    if not ok:
        raise RuntimeError(f"failed to write file: {output_path}")

    print(f"Mask generated: {output_path}")
    print(
        "params: "
        f"size=({args.width}x{args.height}), "
        f"top={args.top_ratio}, bottom={args.bottom_ratio}, "
        f"left={args.left_ratio}, right={args.right_ratio}"
    )


if __name__ == "__main__":
    main()
