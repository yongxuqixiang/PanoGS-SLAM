import subprocess
import sys

# List of packages to install
packages = [
    "opencv-python==4.8.1.78",
    "munch",
    "trimesh",
    "evo==1.11.0",
    "open3d==0.17.0",
    "torchmetrics",
    "imgviz",
    "PyOpenGL",
    "glfw",
    "PyGLM",
    "wandb",
    "lpips",
    "rich",
    "ruff"
]

# Install each package
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"Installed: {package}")
