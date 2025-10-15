# coding: utf-8

# ===============================================================
# Dynamic SAM2 Path Resolver
# ===============================================================
import os
import sys

# Récupère le chemin absolu du dossier où se trouve ce script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Remonte d’un dossier pour atteindre "Documents"
parent_dir = os.path.dirname(current_dir)

# Construit le chemin vers le dossier "sam2"
sam2_path = os.path.join(parent_dir, "sam2")

# Vérifie que le dossier existe et l’ajoute au path
if os.path.exists(sam2_path):
    if sam2_path not in sys.path:
        sys.path.append(sam2_path)
    print(f"[INFO] Module 'sam2' ajouté dynamiquement : {sam2_path}")
else:
    print(f"[ERREUR] Dossier 'sam2' introuvable à l’emplacement : {sam2_path}")
    print("Veuillez vérifier la structure des dossiers.")
    sys.exit(1)

# ===============================================================
# Standard imports
# ===============================================================
import pathlib
import glob
import os

# External imports
import numpy as np
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sv_ttk


# ----------------------------- Model -----------------------------
def build_model(modelname, device):
    configs = {
        "sam2.1-hiera-tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1-hiera-small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1-hiera-base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1-hiera-large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    checkpoints = {
        "sam2.1-hiera-tiny": "checkpoints/sam2.1_hiera_tiny.pt",
        "sam2.1-hiera-small": "checkpoints/sam2.1_hiera_small.pt",
        "sam2.1-hiera-base-plus": "checkpoints/sam2.1_hiera_base_plus.pt",
        "sam2.1-hiera-large": "checkpoints/sam2.1_hiera_large.pt",
    }

    cfg, ckpt = configs[modelname], checkpoints[modelname]
    predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device))
    return predictor

# ----------------------------- File Browser -----------------------------
class FileBrowser:
    def __init__(self, directory, device):
        self.files = []
        exts = ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
        for ext in exts:
            self.files.extend(glob.glob(os.path.join(directory, ext)))
        if len(self.files) == 0:
            raise ValueError(f"No images found in {directory} with extensions {', '.join(exts)}")
        self.directory = directory
        self.device = device

    def __getitem__(self, index):
        image = Image.open(self.files[index])
        image = np.array(image.convert("RGB"))
        return image

    def __len__(self):
        return len(self.files)

# ----------------------------- Segmentation GUI -----------------------------
class BioticSegmentation:
    def __init__(self, directory, modelname):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
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

        self.predictor = build_model(modelname, self.device)
        self.output_path = pathlib.Path("./masks")
        self.image_idx = 0

        # Load images
        self.load_images(directory)
        self.init_ui()

    # ------------------------- Image loading -------------------------
    def load_images(self, directory=None):
        refresh = False
        if directory is None:
            refresh = True
            directory = tkinter.filedialog.askdirectory(initialdir=".", title="Select a directory of images")
            if not directory:
                return
        self.output_path = pathlib.Path("./masks")
        self.image_dataset = FileBrowser(directory, self.device)
        self.out_mask = [None] * len(self.image_dataset)
        self.current_annotation_id = [1] * len(self.image_dataset)
        self.init_processing()

        if refresh:
            self.image_slider.configure(to=len(self.image_dataset) - 1)
            self.update_display()

    def init_processing(self):
        self.current_image = self.image_dataset[self.image_idx]
        self.predictor.set_image(self.current_image)
        self.prompts = [{"positive": [], "negative": [], "box": None} for _ in range(len(self.image_dataset))]
        self.box1_x = self.box1_y = self.box2_x = self.box2_y = None

    # ------------------------- Prediction -------------------------
    def compute_predictions(self):
        if self.out_mask[self.image_idx] is None:
            self.out_mask[self.image_idx] = np.zeros(self.current_image.shape[:2], dtype=np.int32)
        prev_mask = self.out_mask[self.image_idx] == self.current_annotation_id[self.image_idx]
        self.out_mask[self.image_idx][prev_mask] = 0

        ann_obj_id = self.current_annotation_id[self.image_idx]
        image_prompts = {}
        positive_points = self.prompts[self.image_idx]["positive"]
        negative_points = self.prompts[self.image_idx]["negative"]
        box = self.prompts[self.image_idx]["box"]

        points = np.array(positive_points + negative_points, dtype=np.float32)
        if len(points) != 0:
            labels = np.array([1]*len(positive_points) + [0]*len(negative_points), dtype=np.int32)
            image_prompts["point_coords"] = points
            image_prompts["point_labels"] = labels

        if box is not None:
            x1, y1, x2, y2 = box
            image_prompts["box"] = [x1, y1, x2, y2]

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(**image_prompts, multimask_output=True)
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind[0]]

        bkgrd_mask = self.out_mask[self.image_idx] == 0
        self.out_mask[self.image_idx][np.logical_and(bkgrd_mask, masks==1)] = ann_obj_id

    # ------------------------- Coordinate conversion -------------------------
    def image_coords_to_mask_coords(self, x, y):
        x = ((x - self.upper_left_xy[0]) / (self.bottom_right_xy[0] - self.upper_left_xy[0]) * self.current_image.shape[1])
        y = ((y - self.upper_left_xy[1]) / (self.bottom_right_xy[1] - self.upper_left_xy[1]) * self.current_image.shape[0])
        return x, y

    def mask_coords_to_image_coords(self, x, y):
        x = (x / self.current_image.shape[1]) * (self.bottom_right_xy[0] - self.upper_left_xy[0]) + self.upper_left_xy[0]
        y = (y / self.current_image.shape[0]) * (self.bottom_right_xy[1] - self.upper_left_xy[1]) + self.upper_left_xy[1]
        return x, y

    def add_point(self, xy):
        x, y = xy
        x, y = self.image_coords_to_mask_coords(x, y)
        if x < 0 or x >= self.current_image.shape[1] or y < 0 or y >= self.current_image.shape[0]:
            print("Invalid point, outside of the bounds of the image")
            return
        if self.point_mode == "positive":
            self.prompts[self.image_idx]["positive"].append([x, y])
        else:
            self.prompts[self.image_idx]["negative"].append([x, y])

    # ------------------------- Mouse events -------------------------
    def on_mouse_press(self, event):
        b1_x, b1_y = self.image_coords_to_mask_coords(event.x, event.y)
        if b1_x < 0 or b1_x >= self.current_image.shape[1] or b1_y < 0 or b1_y >= self.current_image.shape[0]:
            print("Invalid point, outside of the bounds of the image")
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

    # ------------------------- Keyboard events -------------------------
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
        elif event.keysym == "Next":
            self.change_image(max(0, self.image_idx - 10))
            self.image_slider.set(self.image_idx)
        elif event.keysym == "Prior":
            self.change_image(min(len(self.image_dataset) - 1, self.image_idx + 10))
            self.image_slider.set(self.image_idx)
        elif event.char == "r":
            self.prompts[self.image_idx] = {"positive": [], "negative": [], "box": None}
            self.box1_x = self.box1_y = self.box2_x = self.box2_y = None
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
            self.out_mask[self.image_idx][self.out_mask[self.image_idx] == self.current_annotation_id[self.image_idx]] = 0
            self.update_display()

    # ------------------------- Image navigation -------------------------
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
            resized_img = img.resize((new_width, canvas_height))
            pad = (canvas_width - new_width) // 2
            img = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
            img.paste(resized_img, (pad, 0))
            self.upper_left_xy = (pad, 0)
            self.bottom_right_xy = (pad + new_width, canvas_height)
        else:
            new_height = int(canvas_width / image_aspect_ratio)
            if new_height == 0: new_height = canvas_height
            resized_img = img.resize((canvas_width, new_height))
            pad = (canvas_height - new_height) // 2
            img = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
            img.paste(resized_img, (0, pad))
            self.upper_left_xy = (0, pad)
            self.bottom_right_xy = (canvas_width, pad + new_height)

        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

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
            self.canvas.create_oval(x - self.point_size // 2, y - self.point_size // 2,
                                    x + self.point_size // 2, y + self.point_size // 2,
                                    fill=self.positive_color)
        for x, y in negative_points:
            x, y = self.mask_coords_to_image_coords(x, y)
            self.canvas.create_oval(x - self.point_size // 2, y - self.point_size // 2,
                                    x + self.point_size // 2, y + self.point_size // 2,
                                    fill=self.negative_color)

    # ------------------------- Save masks -------------------------
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
        self.root.title("X-Ray Segmentation")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(main_frame, width=512, height=512)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<Configure>", self.update_display)
        self.root.bind("<Key>", self.on_keystroke)

        image_frame = tk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.image_label = tk.Label(image_frame, text=f"Image: {self.image_idx + 1}/{len(self.image_dataset)}")
        self.image_label.pack(padx=5, pady=5)
        self.image_slider = ttk.Scale(image_frame, from_=0, to=len(self.image_dataset)-1,
                                      orient=tk.HORIZONTAL, command=self.change_image)
        self.image_slider.set(self.image_idx)
        self.image_slider.pack(padx=5, pady=5)

        draw_mode_frame = ttk.Frame(self.root)
        draw_mode_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.draw_mode_label = tk.Label(draw_mode_frame, text=f"Draw Mode:\n{self.draw_mode.capitalize()}", width=20)
        self.draw_mode_label.pack(padx=5, pady=5)

        self.save_button = ttk.Button(self.root, text="Save Masks", command=self.save_masks)
        self.save_button.pack(side=tk.TOP, pady=5)
        self.load_button = ttk.Button(self.root, text="Open directory", command=lambda: self.load_images())
        self.load_button.pack(side=tk.TOP, pady=5)
        self.help_button = ttk.Button(self.root, text="Help", command=self.show_help)
        self.help_button.pack(side=tk.TOP, pady=5)
        self.quit_button = ttk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(side=tk.TOP)

        sv_ttk.set_theme("light")
        self.root.protocol("WM_DELETE_WINDOW", sys.exit)
        self.root.mainloop()

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_text = tk.Text(help_window, wrap=tk.WORD, width=50, height=40)
        help_text.insert(tk.END, "Instructions:\n\n")
        help_text.insert(tk.END, "1. Use the slider to navigate through images.\n")
        help_text.insert(tk.END, "2. Use the 'p' key to switch to positive point mode.\n")
        help_text.insert(tk.END, "3. Use the 'n' key to switch to negative point mode.\n")
        help_text.insert(tk.END, "4. Use the 'b' key to switch to box mode.\n")
        help_text.insert(tk.END, "5. Left/Right arrows to switch images.\n")
        help_text.insert(tk.END, "6. 'r' to reset the current image.\n")
        help_text.insert(tk.END, "7. 'q' to quit.\n")
        help_text.insert(tk.END, "8. 'Return' to increment object ID.\n")
        help_text.insert(tk.END, "9. 'Backspace' to remove last annotation.\n")
        help_text.pack()

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gui.py <directory> <modelname>")
        sys.exit(1)

    directory = sys.argv[1]
    modelname = sys.argv[2]
    BioticSegmentation(directory, modelname)
