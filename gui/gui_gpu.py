# coding: utf-8
"""
SAMourAI - GPU Version
Optimized for workstations with CUDA GPU
Default model: sam2.1-hiera-large (best quality)
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

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ----------------------------- Configuration GPU -----------------------------
DEFAULT_MODEL = "sam2.1-hiera-large"
MAX_IMAGE_SIZE = 2048
AVAILABLE_MODELS = {
    "sam2.1-hiera-large": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "description": "Large (Best quality)"
    },
    "sam2.1-hiera-base-plus": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_base_plus.pt",
        "description": "Base+ (High quality)"
    },
    "sam2.1-hiera-small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_small.pt",
        "description": "Small (Fast)"
    },
    "sam2.1-hiera-tiny": {  # <-- ajout du modèle tiny
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
        "description": "Tiny (Very fast / Low VRAM)"
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


# ----------------------------- File Browser -----------------------------
class FileBrowser:
    def __init__(self, directory, device, max_size=MAX_IMAGE_SIZE):
        self.directory = directory
        self.device = device
        self.max_size = max_size
        self.files = []

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"[INFO] Le dossier {self.directory} n'existait pas, il a été créé.")

        exts = ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
        for ext in exts:
            self.files.extend(glob.glob(os.path.join(self.directory, ext)))

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

    def __getitem__(self, index):
        image = Image.open(self.files[index])
        image = image.convert("RGB")
        
        # Redimensionne si nécessaire
        if max(image.size) > self.max_size:
            ratio = self.max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return np.array(image)

    def __len__(self):
        return len(self.files)


# ----------------------------- Segmentation GUI -----------------------------
class BioticSegmentation:
    def __init__(self, directory, modelname=DEFAULT_MODEL):
        # Vérification GPU
        if not torch.cuda.is_available():
            print("[WARNING] ⚠️  No CUDA GPU detected!")
            print("[WARNING] This version is optimized for GPU.")
            print("[WARNING] Consider using SAMourAI_CPU.py for better performance.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device.upper()}")
        
        if self.device == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
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

        self.predictor = build_model(self.current_modelname, self.device)
        self.output_path = pathlib.Path("./masks")
        self.image_idx = 0

        self.load_images(directory)
        self.init_ui()

    # ------------------------- Model Management -------------------------
    def change_model(self, modelname):
        if modelname == self.current_modelname:
            return
        
        print(f"[INFO] Changing model to {modelname}...")
        self.status_label.config(text="Loading model...")
        self.root.update()
        
        try:
            del self.predictor
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
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
                initialdir=".", 
                title="Select a directory of images"
            )
            if not directory:
                return

        self.image_dataset = FileBrowser(directory, self.device)
        
        if len(self.image_dataset) == 0:
            print("[INFO] Aucun fichier à traiter.")
            return

        self.out_mask = [None] * len(self.image_dataset)
        self.current_annotation_id = [1] * len(self.image_dataset)
        self.init_processing()

    def init_processing(self):
        self.current_image = self.image_dataset[self.image_idx]
        self.predictor.set_image(self.current_image)
        self.prompts = [{"positive": [], "negative": [], "box": None} 
                       for _ in range(len(self.image_dataset))]
        self.box1_x = self.box1_y = self.box2_x = self.box2_y = None

    # ------------------------- Prediction -------------------------
    def compute_predictions(self):
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

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
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
            self.compute_predictions()
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

    # ------------------------- Image Navigation -------------------------
    def change_image(self, value):
        new_idx = int(float(value))
        if new_idx == self.image_idx:
            return
        self.image_idx = new_idx
        self.image_label.config(text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}")
        self.init_processing()
        self.update_display()

    # ------------------------- Display -------------------------
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
        
        if canvas_aspect_ratio > image_aspect_ratio:
            new_width = int(canvas_height * image_aspect_ratio)
            if new_width == 0: new_width = canvas_width
            resized_img = img.resize((new_width, canvas_height), Image.Resampling.LANCZOS)
            pad = (canvas_width - new_width) // 2
            img = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
            img.paste(resized_img, (pad, 0))
            self.upper_left_xy = (pad, 0)
            self.bottom_right_xy = (pad + new_width, canvas_height)
        else:
            new_height = int(canvas_width / image_aspect_ratio)
            if new_height == 0: new_height = canvas_height
            resized_img = img.resize((canvas_width, new_height), Image.Resampling.LANCZOS)
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
        self.root.title("SAMourAI GPU - Image Segmentation Tool")

        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")
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
        image_frame = tk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.image_label = tk.Label(
            image_frame, 
            text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}"
        )
        self.image_label.pack(padx=5, pady=5)
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
            text=f"Device: {self.device.upper()}", 
            fg="green" if self.device == "cuda" else "orange"
        )
        device_label.pack(side=tk.TOP, pady=2)
        
        self.status_label = tk.Label(
            status_frame, 
            text=f"Model: {self.current_modelname}", 
            fg="blue"
        )
        self.status_label.pack(side=tk.TOP, pady=2)

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
        
        self.quit_button = ttk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(side=tk.TOP)

        sv_ttk.set_theme("dark")
        self.root.protocol("WM_DELETE_WINDOW", sys.exit)
        self.root.mainloop()

    def on_model_selected(self, event=None):
        selected = self.model_dropdown.get()
        modelname = selected.split(" - ")[0]
        self.change_model(modelname)

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - GPU Version")
        help_text = tk.Text(help_window, wrap=tk.WORD, width=50, height=30)
        help_text.insert(tk.END, "SAMourAI - GPU Version\n\n")
        help_text.insert(tk.END, "Keyboard shortcuts:\n\n")
        help_text.insert(tk.END, "p - Positive point mode\n")
        help_text.insert(tk.END, "n - Negative point mode\n")
        help_text.insert(tk.END, "b - Box mode\n")
        help_text.insert(tk.END, "←/→ - Previous/Next image\n")
        help_text.insert(tk.END, "r - Reset current image\n")
        help_text.insert(tk.END, "Enter - New object (increment ID)\n")
        help_text.insert(tk.END, "Backspace - Remove last annotation\n")
        help_text.insert(tk.END, "q - Quit\n\n")
        help_text.insert(tk.END, f"GPU: {torch.cuda.get_device_name(0) if self.device == 'cuda' else 'N/A'}\n")
        help_text.insert(tk.END, f"Model: {self.current_modelname}\n")
        help_text.pack()


# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("╔════════════════════════════════════════╗")
        print("║     SAMourAI - GPU Version             ║")
        print("║     Optimized for CUDA workstations    ║")
        print("╚════════════════════════════════════════╝")
        print(f"\nDefault model: {DEFAULT_MODEL}")
        print("\nUsage:")
        print("  python SAMourAI_GPU.py <directory> [model]")
        print("\nExamples:")
        print("  python SAMourAI_GPU.py ./images")
        print("  python SAMourAI_GPU.py ./images sam2.1-hiera-base-plus")
        print("\nAvailable models:")
        for name, info in AVAILABLE_MODELS.items():
            print(f"  • {name}: {info['description']}")
        sys.exit(1)
    
    directory = sys.argv[1]
    modelname = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL
    
    if modelname not in AVAILABLE_MODELS:
        print(f"[ERROR] Unknown model: {modelname}")
        print(f"Available: {list(AVAILABLE_MODELS.keys())}")
        sys.exit(1)
    
    BioticSegmentation(directory, modelname)