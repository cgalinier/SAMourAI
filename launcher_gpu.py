import os
import sys

# --- Path management for PyInstaller ---
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # temporary PyInstaller path
else:
    base_path = os.path.abspath(".")

# --- Main directories ---
project_path = base_path
image_dir = os.path.join(project_path, "image_dir")
assets_dir = os.path.join(project_path, "assets")
configs_dir = os.path.join(project_path, "segment-anything-2", "sam2", "configs")
model_name = "sam2.1-hiera-large"

# --- Automatically create image_dir ---
os.makedirs(image_dir, exist_ok=True)

# --- Add paths for dynamic imports ---
sys.path.insert(0, project_path)

# --- Import the GUI module ---
try:
    from gui_gpu import BioticSegmentation
except ImportError:
    print(f"Error: unable to find gui.py in {project_path}")
    sys.exit(1)

# --- Launch the interface ---
print("Launching SAMourAI...")
BioticSegmentation(image_dir, model_name)
