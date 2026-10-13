"""
Convert a SynPano ground-truth file into the PanoGS-SLAM JSON format.

Input GT format (per line):
    frame_id frame_name x y z qx qy qz qw   (C2W pose)

Outputs:
    data_extrinsics_sorted.json  (camera extrinsics: rotation + center)
    data_views_sorted.json       (view list: filename + pose id)

Usage:
    python tools/format.py --gt gt.txt --output-dir datasets/<scene> \
        --width 1024 --height 512
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, help="input ground-truth txt file")
    parser.add_argument("--output-dir", required=True,
                        help="directory to write data_extrinsics_sorted.json / data_views_sorted.json")
    parser.add_argument("--width", type=int, default=1024, help="panorama width")
    parser.add_argument("--height", type=int, default=512, help="panorama height")
    args = parser.parse_args()

    views = []
    extrinsics = []

    with open(args.gt, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split()
            if len(parts) < 9:
                print(f"[warning] invalid line (expected >= 9 fields): {line.strip()}")
                continue
            idx = int(parts[0])
            # GT format: frame_id frame_name x y z qx qy qz qw
            filename = parts[1]
            tx, ty, tz = map(float, parts[2:5])
            qx, qy, qz, qw = map(float, parts[5:9])

            # Quaternion -> rotation matrix (C2W) -> W2C
            R_c2w = R.from_quat([qx, qy, qz, qw]).as_matrix()
            R_w2c = R_c2w.T
            t_c2w = np.array([tx, ty, tz])

            extrinsics.append({
                "key": idx,
                "value": {
                    "rotation": R_w2c.tolist(),
                    "center": t_c2w.tolist()
                }
            })

            views.append({
                "key": idx,
                "value": {
                    "polymorphic_id": 1073741824,
                    "ptr_wrapper": {
                        "id": 2147483675 + idx,
                        "data": {
                            "local_path": "",
                            "filename": filename,
                            "width": args.width,
                            "height": args.height,
                            "id_view": idx,
                            "id_intrinsic": 0,
                            "id_pose": idx
                        }
                    }
                }
            })

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "data_extrinsics_sorted.json"), "w") as f:
        json.dump({"extrinsics": extrinsics}, f, indent=4)
    with open(os.path.join(args.output_dir, "data_views_sorted.json"), "w") as f:
        json.dump({"views": views}, f, indent=4)

    print(f"Wrote {len(extrinsics)} poses to {args.output_dir}")


if __name__ == "__main__":
    main()
