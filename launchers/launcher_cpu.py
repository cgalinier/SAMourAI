# coding: utf-8
"""
SAMourAI - CPU Version
Default model: sam2.1-hiera-tiny
"""

import os
import sys

# --- Base project path (racine SAMourAI) ---
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"[DEBUG] project_path = {project_path}")

# --- Directories ---
image_dir = os.path.join(project_path, "image_dir")
assets_dir = os.path.join(project_path, "assets")
configs_dir = os.path.join(project_path, "segment-anything-2", "sam2", "configs", "sam2.1")
model_name = "sam2.1-hiera-tiny"
print(f"[DEBUG] configs_dir = {configs_dir}")

# --- Create image_dir if missing ---
os.makedirs(image_dir, exist_ok=True)

# --- Add project root to sys.path for imports ---
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# --- Import GUI module ---
try:
    from gui.gui_cpu import BioticSegmentation
except ImportError as e:
    print(f"[ERREUR] Impossible de trouver gui_cpu.py dans {project_path}/gui")
    print("Détail de l'erreur :", e)
    sys.exit(1)

# --- Verify sam2 configs exist ---
if not os.path.exists(configs_dir):
    print(f"[ERREUR] Dossier 'sam2.1/configs' introuvable à l'emplacement : {configs_dir}")
    sys.exit(1)

# --- Launch GUI ---
print("Launching SAMourAI (CPU)...")
BioticSegmentation(image_dir, model_name)
