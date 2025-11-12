# coding: utf-8
"""
SAMourAI - CPU Version
Optimized for laptops without GPU
Default model: sam2.1-hiera-tiny (fastest on CPU)
"""

# ===============================================================
# Dynamic SAM2 Path Resolver
# ===============================================================
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))  # launchers/
project_root = os.path.dirname(current_dir)  # remonte à SAMourAI/
segment_anything_path = os.path.join(project_root, "segment-anything-2")

if segment_anything_path not in sys.path:
    sys.path.insert(0, segment_anything_path)

# Test import
try:
    import sam2
    print(f"[INFO] Module 'sam2' accessible depuis : {segment_anything_path}")
except ImportError as e:
    print(f"[ERREUR] Impossible d'importer 'sam2' depuis {segment_anything_path}")
    raise

# ===============================================================
# Standard imports
# ===============================================================
import pathlib
import glob
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import torch
import sv_ttk
from threading import Thread
from queue import Queue
import time

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ----------------------------- Configuration CPU -----------------------------
DEFAULT_MODEL = "sam2.1-hiera-tiny"
MAX_IMAGE_SIZE = 2048
PREDICTION_COOLDOWN = 0.3  # Secondes entre prédictions

AVAILABLE_MODELS = {
    "sam2.1-hiera-tiny": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
        "description": "Tiny (Fastest, RECOMMENDED)"
    },
    "sam2.1-hiera-small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_small.pt",
        "description": "Small (Slower but better quality)"
    },
}


# ----------------------------- Model -----------------------------
def build_model(modelname, device):
    if modelname not in AVAILABLE_MODELS:
        raise ValueError(f"Model {modelname} not found")
    
    model_info = AVAILABLE_MODELS[modelname]
    cfg = model_info["config"]
    ckpt = model_info["checkpoint"]
    
    print(f"[INFO] Loading model: {modelname} - {model_info['description']}")
    predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device))
    return predictor


# ----------------------------- File Browser with Preloading -------------------------
class FileBrowser:
    def __init__(self, directory, device, max_size=MAX_IMAGE_SIZE):
        self.directory = directory
        self.device = device
        self.max_size = max_size
        self.files = []
        self.image_cache = {}
        self.preload_queue = Queue()

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"[INFO] Le dossier {self.directory} n'existait pas, il a été créé.")

        self.files = []  
        exts = ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
        for ext in exts:
            self.files.extend(glob.glob(os.path.join(self.directory, ext)))
        
        self.files = list(set(self.files))
        self.files.sort()

        if not self.files:
            print(f"[INFO] Aucune image trouvée. Sélection d'images...")
            self.select_images()

    def select_images(self):
        files = tkinter.filedialog.askopenfilenames(
            title="Sélectionnez des images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")]
        )
        if files:
            self.files = list(files)
            print(f"[INFO] {len(self.files)} images sélectionnées.")
        else:
            print("[INFO] Aucune image sélectionnée.")

    def load_and_resize_image(self, index):
        """Charge et redimensionne l'image pour économiser la mémoire"""
        if index in self.image_cache:
            return self.image_cache[index]
        
        image = Image.open(self.files[index])
        image = image.convert("RGB")
        
        # Redimensionne agressivement pour CPU
        if max(image.size) > self.max_size:
            ratio = self.max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        image_array = np.array(image)
        self.image_cache[index] = image_array
        return image_array

    def preload_images(self, indices):
        """Précharge les images en arrière-plan"""
        for idx in indices:
            if idx not in self.image_cache and 0 <= idx < len(self.files):
                self.preload_queue.put(idx)

    def __getitem__(self, index):
        return self.load_and_resize_image(index)

    def __len__(self):
        return len(self.files)


# ----------------------------- Segmentation GUI -------------------------
class BioticSegmentation:
    def __init__(self, directory, modelname=DEFAULT_MODEL):
        # Force CPU mode
        self.device = "cpu"
        print(f"[INFO] SAMourAI CPU Version - Optimized for laptops")
        print(f"[INFO] Using device: CPU")
        
        # Optimisations CPU critiques
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        print("[INFO] CPU optimizations applied")
        
        if torch.cuda.is_available():
            print("[INFO] ⚡ GPU detected but using CPU version")
            print("[INFO] Consider using SAMourAI_GPU.py for better performance")
        
        self.current_modelname = modelname
        self.point_size = 10
        self.last_mask_color = np.array([0, 0, 1])
        self.objects_colors = []
        self.positive_color = "red"
        self.negative_color = "green"
        self.drawing_box = False
        self.draw_mode = "box"
        self.point_mode = "positive"
        self.upper_left_xy = (0, 0)
        self.bottom_right_xy = (0, 0)
        
        # Variables d'optimisation
        self.prediction_pending = False
        self.last_prediction_time = 0
        self.prediction_cooldown = PREDICTION_COOLDOWN
        self.preload_thread = None
        self.preload_running = True

        self.predictor = build_model(self.current_modelname, self.device)
        self.output_path = pathlib.Path("./masks")
        self.image_idx = 0

        self.load_images(directory)
        self.start_preload_thread()
        self.init_ui()

    def get_current_filename(self):
        """Renvoie le nom du fichier actuel, tronqué si trop long"""
        if len(self.image_dataset.files) == 0:
            return ""
        full_name = str(pathlib.Path(self.image_dataset.files[self.image_idx]).name)
        max_len = 60
        if len(full_name) > max_len:
            return full_name[:30] + "…" + full_name[-25:]
        return full_name

    # ------------------------- Preloading Thread -------------------------
    def start_preload_thread(self):
        """Démarre le thread de préchargement d'images"""
        def preload_worker():
            while self.preload_running:
                try:
                    idx = self.image_dataset.preload_queue.get(timeout=1)
                    self.image_dataset.load_and_resize_image(idx)
                    print(f"[INFO] Image {idx} préchargée")
                except:
                    pass
        
        self.preload_thread = Thread(target=preload_worker, daemon=True)
        self.preload_thread.start()

    def preload_adjacent_images(self):
        """Précharge les images adjacentes"""
        indices = []
        for offset in [-2, -1, 1, 2]:
            idx = self.image_idx + offset
            if 0 <= idx < len(self.image_dataset):
                indices.append(idx)
        self.image_dataset.preload_images(indices)

    # ------------------------- Model Management -------------------------
    def change_model(self, modelname):
        if modelname == self.current_modelname:
            return
        
        print(f"[INFO] Changing model to {modelname}...")
        self.status_label.config(text="Loading model... (may take time)")
        self.root.update()
        
        try:
            del self.predictor
            
            self.current_modelname = modelname
            self.predictor = build_model(modelname, self.device)
            
            if len(self.image_dataset) > 0:
                self.predictor.set_image(self.current_image)
            
            self.status_label.config(text=f"Model: {modelname}")
            print(f"[INFO] ✓ Model changed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.status_label.config(text="Error loading model!")

    # ------------------------- Image Loading -------------------------
    
    def load_images(self, directory=None):
        if directory is None:
            directory = tkinter.filedialog.askdirectory(
                initialdir=self.last_dir if hasattr(self, 'last_dir') else ".", 
                title="Select a directory of images"
            )
            if not directory:
                return
            
        # Last open directory
        self.last_dir = directory

        self.image_dataset = FileBrowser(directory, self.device)
        
        if len(self.image_dataset) == 0:
            print("[INFO] Aucun fichier à traiter.")
            return

        self.out_mask = [None] * len(self.image_dataset)
        self.current_annotation_id = [1] * len(self.image_dataset)
        self.init_processing()

        # ------------------ MAJ slider ------------------
        if hasattr(self, 'image_slider'):
            self.image_slider.config(from_=0, to=len(self.image_dataset)-1)
            self.image_slider.set(0)
            self.image_idx = 0
            self.image_label.config(text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}")
            self.filename_label.config(text=self.get_current_filename())
            self.update_display()  # affiche la première image du nouveau dossier


    def init_processing(self):
        self.current_image = self.image_dataset[self.image_idx]
        self.predictor.set_image(self.current_image)
        self.prompts = [{"positive": [], "negative": [], "box": None} 
                        for _ in range(len(self.image_dataset))]
        self.box1_x = self.box1_y = self.box2_x = self.box2_y = None


    # ------------------------- Prediction with Throttling -------------------------
    def compute_predictions(self):
        """Calcule les prédictions avec throttling pour CPU"""
        current_time = time.time()
        
        # Throttling pour éviter surcharge CPU
        if current_time - self.last_prediction_time < self.prediction_cooldown:
            self.prediction_pending = True
            self.root.after(int(self.prediction_cooldown * 1000), self.delayed_prediction)
            return
        
        self.last_prediction_time = current_time
        self.prediction_pending = False
        self._execute_prediction()

    def delayed_prediction(self):
        """Exécute une prédiction retardée si nécessaire"""
        if self.prediction_pending:
            self._execute_prediction()

    def _execute_prediction(self):
        """Exécute réellement la prédiction"""
        if self.out_mask[self.image_idx] is None:
            self.out_mask[self.image_idx] = np.zeros(
                self.current_image.shape[:2], dtype=np.int32
            )
        
        prev_mask = self.out_mask[self.image_idx] == self.current_annotation_id[self.image_idx]
        self.out_mask[self.image_idx][prev_mask] = 0

        ann_obj_id = self.current_annotation_id[self.image_idx]
        image_prompts = {}
        positive_points = self.prompts[self.image_idx]["positive"]
        negative_points = self.prompts[self.image_idx]["negative"]
        box = self.prompts[self.image_idx]["box"]

        points = np.array(positive_points + negative_points, dtype=np.float32)
        if len(points) != 0:
            labels = np.array([1]*len(positive_points) + [0]*len(negative_points), 
                            dtype=np.int32)
            image_prompts["point_coords"] = points
            image_prompts["point_labels"] = labels

        if box is not None:
            x1, y1, x2, y2 = box
            image_prompts["box"] = [x1, y1, x2, y2]

        # Utilise float16 pour CPU
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16):
            masks, scores, logits = self.predictor.predict(
                **image_prompts, 
                multimask_output=True
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind[0]]

        bkgrd_mask = self.out_mask[self.image_idx] == 0
        self.out_mask[self.image_idx][np.logical_and(bkgrd_mask, masks==1)] = ann_obj_id

    # ------------------------- Coordinate Conversion -------------------------
    def image_coords_to_mask_coords(self, x, y):
        x = ((x - self.upper_left_xy[0]) / (self.bottom_right_xy[0] - self.upper_left_xy[0]) 
             * self.current_image.shape[1])
        y = ((y - self.upper_left_xy[1]) / (self.bottom_right_xy[1] - self.upper_left_xy[1]) 
             * self.current_image.shape[0])
        return x, y

    def mask_coords_to_image_coords(self, x, y):
        x = (x / self.current_image.shape[1]) * (self.bottom_right_xy[0] - self.upper_left_xy[0]) + self.upper_left_xy[0]
        y = (y / self.current_image.shape[0]) * (self.bottom_right_xy[1] - self.upper_left_xy[1]) + self.upper_left_xy[1]
        return x, y

    def add_point(self, xy):
        x, y = xy
        x, y = self.image_coords_to_mask_coords(x, y)
        if x < 0 or x >= self.current_image.shape[1] or y < 0 or y >= self.current_image.shape[0]:
            return
        if self.point_mode == "positive":
            self.prompts[self.image_idx]["positive"].append([x, y])
        else:
            self.prompts[self.image_idx]["negative"].append([x, y])

    # ------------------------- Mouse Events -------------------------
    def on_mouse_press(self, event):
        b1_x, b1_y = self.image_coords_to_mask_coords(event.x, event.y)
        if (b1_x < 0 or b1_x >= self.current_image.shape[1] or 
            b1_y < 0 or b1_y >= self.current_image.shape[0]):
            return
        if self.draw_mode == "box":
            self.box1_x, self.box1_y = b1_x, b1_y
            self.drawing_box = True
        else:
            self.add_point((event.x, event.y))
        self.compute_predictions()
        self.update_display()

    def on_mouse_drag(self, event):
        if self.draw_mode == "box" and self.drawing_box:
            self.box2_x, self.box2_y = self.image_coords_to_mask_coords(event.x, event.y)
            self.prompts[self.image_idx]["box"] = [
                min(self.box1_x, self.box2_x),
                min(self.box1_y, self.box2_y),
                max(self.box1_x, self.box2_x),
                max(self.box1_y, self.box2_y),
            ]
            # Pas de prédiction pendant drag (trop lent sur CPU)
            self.update_display()

    def on_mouse_release(self, event):
        if self.drawing_box:
            self.drawing_box = False
            if self.draw_mode == "box":
                self.compute_predictions()
                self.update_display()

    # ------------------------- Keyboard Events -------------------------
    def on_keystroke(self, event):
        if event.char == "p":
            self.draw_mode = "point"
            self.point_mode = "positive"
            self.draw_mode_label.config(text="Draw Mode:\nPoint (Positive)")
        elif event.char == "n":
            self.draw_mode = "point"
            self.point_mode = "negative"
            self.draw_mode_label.config(text="Draw Mode:\nPoint (Negative)")
        elif event.char == "b":
            self.draw_mode = "box"
            self.draw_mode_label.config(text="Draw Mode:\nBox")
        elif event.keysym == "Left":
            self.change_image(max(0, self.image_idx - 1))
            self.image_slider.set(self.image_idx)
        elif event.keysym == "Right":
            self.change_image(min(len(self.image_dataset) - 1, self.image_idx + 1))
            self.image_slider.set(self.image_idx)
        elif event.char == "r":
            self.prompts[self.image_idx] = {"positive": [], "negative": [], "box": None}
            self.box1_x = self.box1_y = self.box2_x = self.box2_y = None
            if self.out_mask[self.image_idx] is not None:
                self.out_mask[self.image_idx][...] = 0
            self.current_annotation_id[self.image_idx] = 1
            self.update_display()
        elif event.char == "q":
            self.preload_running = False
            self.root.quit()
        elif event.keysym == "Return":
            self.current_annotation_id[self.image_idx] += 1
            self.prompts[self.image_idx] = {"positive": [], "negative": [], "box": None}
            self.update_display()
        elif event.keysym == "BackSpace":
            self.prompts[self.image_idx] = {"positive": [], "negative": [], "box": None}
            if self.out_mask[self.image_idx] is not None:
                self.out_mask[self.image_idx][
                    self.out_mask[self.image_idx] == self.current_annotation_id[self.image_idx]
                ] = 0
            self.update_display()

        elif event.char == "u":  # Undo last mask
            self.undo_last_mask()

    # ------------------------- Undo Last Mask -------------------------
    def undo_last_mask(self):
        if self.current_annotation_id[self.image_idx] > 1:
            last_id = self.current_annotation_id[self.image_idx] - 1
            mask = self.out_mask[self.image_idx]
            if mask is not None:
                mask[mask == last_id] = 0  # Supprime le dernier objet
            self.current_annotation_id[self.image_idx] -= 1
            self.prompts[self.image_idx] = {"positive": [], "negative": [], "box": None}
            self.update_display()
            print(f"[INFO] Last mask removed. Current annotation ID: {self.current_annotation_id[self.image_idx]}")
        else:
            print("[INFO] No mask to undo.")


    # ------------------------- Image Navigation -------------------------
    def change_image(self, value):
        new_idx = int(float(value))
        if new_idx == self.image_idx:
            return
        self.image_idx = new_idx
        self.image_label.config(text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}")
        self.filename_label.config(text=self.get_current_filename())
        self.init_processing()
        self.preload_adjacent_images()
        self.update_display()

    # ------------------------- Display (Optimized for CPU) -------------------------
    def update_display(self, event=None):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        current_image = self.current_image.copy() / 255.0

        if len(self.objects_colors) < self.current_annotation_id[self.image_idx]:
            self.objects_colors.append(np.random.rand(3))

        overlaid = current_image.copy()
        for i in range(1, self.current_annotation_id[self.image_idx]):
            mask = self.out_mask[self.image_idx] == i
            overlaid[mask] = 0.5 * current_image[mask] + 0.5 * self.objects_colors[i - 1]

        mask = self.out_mask[self.image_idx] == self.current_annotation_id[self.image_idx]
        overlaid[mask] = 0.5 * current_image[mask] + 0.5 * self.last_mask_color
        img = Image.fromarray((overlaid * 255).astype(np.uint8))

        canvas_aspect_ratio = canvas_width / canvas_height
        image_aspect_ratio = img.width / img.height
        
        # NLANCZOS pour CPU 
        resample_method = Image.Resampling.LANCZOS
        
        if canvas_aspect_ratio > image_aspect_ratio:
            new_width = int(canvas_height * image_aspect_ratio)
            if new_width == 0: new_width = canvas_width
            resized_img = img.resize((new_width, canvas_height), resample_method)
            pad = (canvas_width - new_width) // 2
            img = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
            img.paste(resized_img, (pad, 0))
            self.upper_left_xy = (pad, 0)
            self.bottom_right_xy = (pad + new_width, canvas_height)
        else:
            new_height = int(canvas_width / image_aspect_ratio)
            if new_height == 0: new_height = canvas_height
            resized_img = img.resize((canvas_width, new_height), resample_method)
            pad = (canvas_height - new_height) // 2
            img = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
            img.paste(resized_img, (0, pad))
            self.upper_left_xy = (0, pad)
            self.bottom_right_xy = (canvas_width, pad + new_height)

        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        # Draw annotations
        positive_points = self.prompts[self.image_idx]["positive"]
        negative_points = self.prompts[self.image_idx]["negative"]
        box = self.prompts[self.image_idx]["box"]

        if box is not None:
            x1, y1, x2, y2 = box
            x1, y1 = self.mask_coords_to_image_coords(x1, y1)
            x2, y2 = self.mask_coords_to_image_coords(x2, y2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

        for x, y in positive_points:
            x, y = self.mask_coords_to_image_coords(x, y)
            self.canvas.create_oval(
                x - self.point_size // 2, y - self.point_size // 2,
                x + self.point_size // 2, y + self.point_size // 2,
                fill=self.positive_color
            )
        for x, y in negative_points:
            x, y = self.mask_coords_to_image_coords(x, y)
            self.canvas.create_oval(
                x - self.point_size // 2, y - self.point_size // 2,
                x + self.point_size // 2, y + self.point_size // 2,
                fill=self.negative_color
            )

    # ------------------------- Save Masks -------------------------
    def save_masks(self):
        if not self.output_path.exists():
            self.output_path.mkdir()
        for src_file, mask in zip(self.image_dataset.files, self.out_mask):
            if mask is not None:
                mask_img = Image.fromarray(mask.astype(np.uint8))
                src_file = pathlib.Path(src_file)
                output_file = str(self.output_path / src_file.stem) + ".png"
                print(f"Saving {output_file}")
                mask_img.save(output_file)

    # ------------------------- UI -------------------------
    def init_ui(self):
        self.root = tk.Tk()
        self.root.title("SAMourAI CPU - Laptop Optimized")

        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icon_l.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconphoto(True, ImageTk.PhotoImage(Image.open(icon_path)))
            except:
                pass


        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(main_frame, width=512, height=512)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Configure>", self.update_display)
        self.root.bind("<Key>", self.on_keystroke)

        # Image navigation
        from pathlib import Path

        image_frame = tk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Label "1/20"
        self.image_label = tk.Label(
            image_frame, 
            text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}"
        )
        self.image_label.pack(padx=5, pady=2)

        # Label pour nom de fichier complet (tronqué si trop long)
        self.filename_label = tk.Label(
            image_frame,
            text=self.get_current_filename(),
            wraplength=500,   # largeur max avant retour à la ligne
            fg="lightgray",
            justify=tk.LEFT
        )
        self.filename_label.pack(padx=5, pady=2)

        # Slider
        self.image_slider = ttk.Scale(
            image_frame, 
            from_=0, 
            to=len(self.image_dataset)-1,
            orient=tk.HORIZONTAL, 
            command=self.change_image
        )
        self.image_slider.set(self.image_idx)
        self.image_slider.pack(padx=5, pady=5)


        # Draw mode
        draw_mode_frame = ttk.Frame(self.root)
        draw_mode_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.draw_mode_label = tk.Label(
            draw_mode_frame, 
            text=f"Draw Mode:\n{self.draw_mode.capitalize()}", 
            width=20
        )
        self.draw_mode_label.pack(padx=5, pady=5)

        # Status
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        device_label = tk.Label(
            status_frame, 
            text="Device: CPU (Optimized)", 
            fg="orange"
        )
        device_label.pack(side=tk.TOP, pady=2)
        
        self.status_label = tk.Label(
            status_frame, 
            text=f"Model: {self.current_modelname}", 
            fg="blue"
        )
        self.status_label.pack(side=tk.TOP, pady=2)

        # ------------------------- Draw mode label -------------------------
        self.draw_mode_label = tk.Label(
            status_frame,
            text=f"Draw Mode:\n{self.draw_mode}",
            fg="yellow"
        )
        self.draw_mode_label.pack(side=tk.TOP, pady=2)

        # Model selector
        model_frame = ttk.Frame(self.root)
        model_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Label(model_frame, text="Model:").pack(side=tk.TOP, pady=2)
        
        self.model_var = tk.StringVar(value=self.current_modelname)
        model_options = [f"{name} - {info['description']}" 
                        for name, info in AVAILABLE_MODELS.items()]
        
        self.model_dropdown = ttk.Combobox(
            model_frame, 
            textvariable=self.model_var,
            values=model_options,
            state="readonly",
            width=35
        )
        self.model_dropdown.pack(side=tk.TOP, padx=5, pady=2)
        self.model_dropdown.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        current_display = f"{self.current_modelname} - {AVAILABLE_MODELS[self.current_modelname]['description']}"
        self.model_dropdown.set(current_display)

        # CPU Tips
        tips_label = tk.Label(
            self.root, 
            text="💡 CPU Mode: Predictions may be slower",
            fg="gray",
            font=("Arial", 8)
        )
        tips_label.pack(side=tk.TOP, pady=2)

        # Buttons
        self.save_button = ttk.Button(self.root, text="Save Masks", command=self.save_masks)
        self.save_button.pack(side=tk.TOP, pady=5)
        
        self.load_button = ttk.Button(
            self.root, 
            text="Open directory", 
            command=lambda: self.load_images()
        )
        self.load_button.pack(side=tk.TOP, pady=5)
        
        self.help_button = ttk.Button(self.root, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.TOP, pady=5)
        
        self.quit_button = ttk.Button(self.root, text="Quit", command=self.on_close)
        self.quit_button.pack(side=tk.TOP)

        sv_ttk.set_theme("dark")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        """Fermeture propre"""
        self.preload_running = False
        if self.preload_thread:
            self.preload_thread.join(timeout=1)
        sys.exit()

    def on_model_selected(self, event=None):
        selected = self.model_dropdown.get()
        modelname = selected.split(" - ")[0]
        
        if modelname == "sam2.1-hiera-small":
            from tkinter import messagebox
            result = messagebox.askyesno(
                "Performance Warning", 
                "Small model is significantly slower on CPU.\n\n"
                "Expect 3-5x longer processing times.\n\n"
                "Continue anyway?",
                icon='warning'
            )
            if not result:
                current_display = f"{self.current_modelname} - {AVAILABLE_MODELS[self.current_modelname]['description']}"
                self.model_dropdown.set(current_display)
                return
        
        self.change_model(modelname)

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - CPU Version")
        help_text = tk.Text(help_window, wrap=tk.WORD, width=55, height=35)
        help_text.insert(tk.END, "SAMourAI - CPU Version (Laptop Optimized)\n\n")
        help_text.insert(tk.END, "Keyboard shortcuts:\n\n")
        help_text.insert(tk.END, "p - Positive point mode\n")
        help_text.insert(tk.END, "n - Negative point mode\n")
        help_text.insert(tk.END, "b - Box mode\n")
        help_text.insert(tk.END, "←/→ - Previous/Next image\n")
        help_text.insert(tk.END, "r - Reset current image\n")
        help_text.insert(tk.END, "Enter - New object (increment ID)\n")
        help_text.insert(tk.END, "Backspace - Remove last annotation\n")
        help_text.insert(tk.END, "u - Remove last mask object annotation\n")
        help_text.insert(tk.END, "q - Quit\n\n")
        help_text.insert(tk.END, "CPU Optimizations Active:\n\n")
        help_text.insert(tk.END, f"• Images resized to max {MAX_IMAGE_SIZE}px\n")
        help_text.insert(tk.END, f"• Prediction throttling ({PREDICTION_COOLDOWN}s cooldown)\n")
        help_text.insert(tk.END, "• Background image preloading\n")
        help_text.insert(tk.END, "• No prediction during box drag\n")
        help_text.insert(tk.END, "• LANCZOS resampling\n")
        help_text.insert(tk.END, "• Memory-efficient caching\n\n")
        help_text.insert(tk.END, f"Current Model: {self.current_modelname}\n")
        help_text.insert(tk.END, f"Max Image Size: {MAX_IMAGE_SIZE}px\n")
        help_text.pack()


# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("╔════════════════════════════════════════╗")
        print("║     SAMourAI - CPU Version             ║")
        print("║     Optimized for laptops without GPU  ║")
        print("╚════════════════════════════════════════╝")
        print(f"\nDefault model: {DEFAULT_MODEL} (fastest)")
        print(f"Max image size: {MAX_IMAGE_SIZE}px (memory efficient)")
        print("\nUsage:")
        print("  python SAMourAI_CPU.py <directory> [model]")
        print("\nExamples:")
        print("  python SAMourAI_CPU.py ./images")
        print("  python SAMourAI_CPU.py ./images sam2.1-hiera-small")
        print("\nAvailable models:")
        for name, info in AVAILABLE_MODELS.items():
            print(f"  • {name}: {info['description']}")
        print("\n💡 Tips for best CPU performance:")
        print("  - Use sam2.1-hiera-tiny (default)")
        print("  - Close other heavy applications")
        print("  - Work with smaller images when possible")
        sys.exit(1)
    
    directory = sys.argv[1]
    modelname = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL
    
    if modelname not in AVAILABLE_MODELS:
        print(f"[ERROR] Unknown model: {modelname}")
        print(f"Available: {list(AVAILABLE_MODELS.keys())}")
        sys.exit(1)
    
    BioticSegmentation(directory, modelname)