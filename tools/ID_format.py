"""
Convert a PALVIO ground-truth file into the PanoGS-SLAM JSON format.

Each panorama frame is matched to the nearest ground-truth pose by timestamp
(PALVIO images are named by timestamp, e.g. 0.058392.png).

Input GT format (per line):
    timestamp tx ty tz qx qy qz qw   (C2W pose)

Outputs:
    data_extrinsics_sorted.json  (camera extrinsics: rotation + center)
    data_views_sorted.json       (view list: filename + pose id)

Usage:
    python tools/ID_format.py --image-dir datasets/ID02/pro_images_test \
        --gt datasets/ID02/groundtruth.txt --output-dir datasets/ID02
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", required=True,
                        help="directory of equirectangular frames named by timestamp")
    parser.add_argument("--gt", required=True, help="input ground-truth txt file")
    parser.add_argument("--output-dir", required=True,
                        help="directory to write data_extrinsics_sorted.json / data_views_sorted.json")
    args = parser.parse_args()

    # Read the ground truth
    gt_data = []
    with open(args.gt, "r") as f:
        for line in f:
            items = list(map(float, line.strip().split()))
            timestamp = items[0]
            translation = items[1:4]
            quaternion = items[4:]
            gt_data.append((timestamp, translation, quaternion))
    gt_data.sort(key=lambda x: x[0])

    # Collect image files sorted by timestamp
    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png")),
                         key=lambda x: float(os.path.splitext(os.path.basename(x))[0]))
    image_info = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        ts = float(os.path.splitext(basename)[0])
        image_info.append((ts, basename))

    extrinsics = {"extrinsics": []}
    views = {"views": []}

    for idx, (img_time, filename) in enumerate(image_info):
        # Match the nearest ground-truth pose
        closest = min(gt_data, key=lambda x: abs(x[0] - img_time))
        _, t, q = closest

        # Quaternion -> rotation matrix; store W2C rotation + C2W center
        rot = R.from_quat(q).as_matrix()
        rot = rot.T
        t = np.array(t).reshape(3, 1)
        center = t.flatten().tolist()

        extrinsics["extrinsics"].append({
            "key": idx + 1,
            "value": {
                "rotation": rot.tolist(),
                "center": center
            }
        })

        views["views"].append({
            "key": idx + 1,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": 2147483675 + idx + 1,
                    "data": {
                        "local_path": "",
                        "filename": filename,
                        "width": 1024,
                        "height": 512,
                        "id_view": idx + 1,
                        "id_intrinsic": 0,
                        "id_pose": idx + 1
                    }
                }
            }
        })

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "data_extrinsics_sorted.json"), "w") as f:
        json.dump(extrinsics, f, indent=4)
    with open(os.path.join(args.output_dir, "data_views_sorted.json"), "w") as f:
        json.dump(views, f, indent=4)

    print(f"Wrote {len(image_info)} poses to {args.output_dir}")


if __name__ == "__main__":
    main()
