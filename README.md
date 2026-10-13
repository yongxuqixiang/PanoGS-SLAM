
<p align="center">
  <h1 align="center">PanoGS-SLAM</h1>
  <h3 align="center">Panoramic 3D Gaussian Splatting SLAM</h3>

  <p align="center">
    <a href="https://yongxuqixiang.github.io/yongqi/"><strong>Yongqi Mao</strong></a>,
    Hao Shi,
    Yufan Zhang,
    Zhonghua Yi,
    Xiangfei Guo,
    Kaiwei Wang
  </p>

  <p align="center">
    <strong>IROS 2026</strong> &nbsp;|&nbsp;
    <a href="https://arxiv.org/abs/2609.17387">Paper (arXiv)</a>
  </p>

  <p align="center">
    <video src="https://github.com/user-attachments/assets/c9a58b97-0ae1-4cf7-a70d-96b524f3628c" controls muted width="860"></video>
  </p>

  <p align="center">
    <a href="https://arxiv.org/abs/2609.17387"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.17387-b31b1b.svg"></a>
    <a href="https://www.iros.org/"><img alt="IROS 2026" src="https://img.shields.io/badge/IROS-2026-6A8759.svg"></a>
    <a href="#license"><img alt="license" src="https://img.shields.io/badge/license-non--commercial-orange.svg"></a>
  </p>
</p>

---

**PanoGS-SLAM** is a real-time SLAM system that reconstructs scenes with 3D Gaussian Splatting from **360° panoramic images**. It is the first panoramic dense SLAM system built on 3D Gaussian Splatting, performing differentiable rendering and pose optimization directly in the spherical domain.

## Installation

Requirements: NVIDIA GPU (CUDA 11.6), [Anaconda](https://www.anaconda.com/).

```bash
# 1. Clone the repository (BiFuse is bundled, no submodules needed)
git clone https://github.com/<your-org>/PanoGS-SLAM.git
cd PanoGS-SLAM

# 2. Create the conda environment
conda env create -f environment.yml
conda activate panogs

# 3. Compile the CUDA rasterizers
pip install submodules/diff-gaussian-rasterization-w-pose
pip install submodules/odgs-gaussian-rasterization
pip install submodules/simple-knn

# 4. Download the BiFuse pretrained weights (~1 GB) for panoramic depth priors
#    from https://github.com/yuhsuanyeh/BiFuse and place it in the repo root as:
#    BiFuse_Pretrained.pkl
```

## Quick Start

A small example sequence (SynPano *meetingroom*, first 80 frames @ 960×480) ships with the repository, so you can verify the installation without downloading any dataset:

```bash
# Run the bundled example (GUI)
python slam.py --config configs/pano/example.yaml

# Run in evaluation mode (no GUI; ATE + rendering metrics, results saved under results/)
python slam.py --config configs/pano/example.yaml --eval
```

> Note: `--eval` enables [Weights & Biases](https://wandb.ai/) logging. If you do not have a wandb account, run with `WANDB_MODE=disabled` to skip it.

## Data Preparation

### Benchmarks

PanoGS-SLAM is evaluated on two panoramic benchmarks:

- **[SynPano](https://github.com/guoxf304/SynPano-Dataset)** — synthetic indoor rooms rendered as 360°×180° equirectangular panoramas (native ERP format).
- **[PALVIO](https://github.com/flysoaryun/LF-VIO)** — real aerial sequences from a panoramic annular lens (PAL) camera (360°×[40°,120°] FoV). The PALVIO sequences need to be converted from native PAL frames to equirectangular panoramas first; `tools/format.py` and `tools/ID_format.py` show the expected JSON formats for the converted data.

### Image resolution

Images are resized by the loader to the `width` / `height` specified in the config, so both full-resolution and pre-downscaled folders work out of the box:

| Benchmark | Native resolution | **Resolution used in our experiments / configs** |
|---|---|---|
| SynPano | 1920 × 960 | **960 × 480** (2× downscale, `images_small/`) |
| PALVIO | 1024 × 512 | **1024 × 512** (native) |

> For SynPano we run all experiments at 960 × 480: we use the provided downscaled `images_small/` folder (or equivalently point `dataset_path_image` to the full-res `images/` — the loader resizes automatically). The bundled `datasets/example` follows the same 960 × 480 setting.

### Expected layout

PanoGS-SLAM consumes panoramic image sequences with extrinsics in the layout below (see `utils/dataset.py: PanoramicDataset`):

```
datasets/<scene>/
├── data_extrinsics_sorted.json    # camera extrinsics (rotation + center)
├── data_views_sorted.json         # view list (filename + pose id)
├── groundtruth.txt                # (optional) GT trajectory for ATE, TUM-like:
│                                  #   # frame_id image_name x y z qx qy qz qw
└── images/                        # equirectangular RGB frames
```

Place each benchmark under `datasets/` (the configs expect `datasets/SynPano/<scene>` and `datasets/PALVIO/<IDxx>`) and point `Dataset.dataset_path` / `dataset_path_image` of the corresponding config to your location.

`Dataset.mask_path` optionally masks out invalid pixels — it is **required for PALVIO** (the PAL camera has ~25% blind-zone pixels; use the bundled `mask/mask.png`). `tools/generate_border_mask.py` can generate masks for other cameras.

Classic benchmark datasets (TUM / Replica / EuRoC) are still supported via the scripts in `scripts/`.

## Running

Each benchmark sequence ships with a ready-to-use config under `configs/pano/`:

```
configs/pano/
├── base_config.yaml            # shared defaults, inherited by all scenes
├── example.yaml                # bundled example sequence (see Quick Start)
├── PALVIO.yaml                 # all PALVIO sequences (ID01–ID10): change dataset_path only
├── SynPano_apartment.yaml      # SynPano room3
├── SynPano_livingroom.yaml     # SynPano room1
├── SynPano_meetingroom.yaml    # SynPano room2
├── SynPano_washroom.yaml       # SynPano room4
└── SynPano_jodo.yaml           # SynPano room5
```

```bash
# PALVIO (e.g. ID02): edit dataset_path/dataset_path_image in PALVIO.yaml
python slam.py --config configs/pano/PALVIO.yaml --eval

# SynPano (e.g. meetingroom)
python slam.py --config configs/pano/SynPano_meetingroom.yaml --eval
```

Results (trajectory, rendering metrics, Gaussian maps) are saved under `results/`.

### Evaluation

- **ATE** is computed automatically at the end of a run (see `utils/eval_utils.py: eval_ate`), or offline with `tools/eval.py` (evo-based).
- **Rendering metrics** (PSNR / SSIM / LPIPS) over the final Gaussian map can also be evaluated offline from saved outputs:

```bash
python tools/eval_panoramic_rendering_metrics.py \
    --traj_file  <results/.../trj_XXXX.json> \
    --ply_file   <results/.../final_after_opt.ply> \
    --image_dir  datasets/<scene>/images \
    --output_dir eval_output \
    --use_viewport_psnr   # 3x 120° pinhole viewports (paper protocol)
```

## Acknowledgements

PanoGS-SLAM is based on the following open-source projects. Please follow their licenses (also see [Dependencies.md](Dependencies.md)):

- [MonoGS](https://github.com/muskie82/MonoGS) — the base real-time monocular Gaussian Splatting SLAM system.
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — Gaussian splatting rasterizer and scene representation (non-commercial license).
- [ODGS](https://github.com/esw0116/odgs) — omnidirectional Gaussian rasterizer adapted for panoramic splatting.
- [BiFuse](https://github.com/yuhsuanyeh/BiFuse) — monocular 360° depth estimation.
- [GaussianSplattingViewer](https://github.com/limacv/GaussianSplattingViewer) — the OpenGL viewer used by the GUI.

## Citation

If you find PanoGS-SLAM useful in your research, please cite:

```bibtex
@inproceedings{mao2026panogsslam,
  title     = {{PanoGS-SLAM: Panoramic 3D Gaussian Splatting SLAM}},
  author    = {Mao, Yongqi and Shi, Hao and Zhang, Yufan and Yi, Zhonghua and Guo, Xiangfei and Wang, Kaiwei},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  note      = {arXiv:2609.17387}
}
```

## License

The code derived from 3D Gaussian Splatting is licensed for **non-commercial, research purposes only** (see `gaussian_splatting/LICENSE.md` and [Dependencies.md](Dependencies.md)). Other components follow their respective licenses.
