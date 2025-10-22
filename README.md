# SAMourAI — Installation and User Guide
*(Based on the [SAMBIOTIC](https://github.com/jeremyfix/sambiotic) repository by Jérémy Fix)*
> **Author:** Corentin GALINIER — University of Lorraine  
> **Creation date:** October 22, 2025

---

**SAMourAI** is a tool for **semi-automatic image segmentation**, built on [**SAM 2**](https://github.com/facebookresearch/sam2), developed by *Meta (Facebook Research)*.  
It provides a **graphical interface** to perform image segmentation, available in both **CPU** and **GPU** modes.

---

![UI](assets/ui.png)

---

## 1. Prerequisites

### Hardware Configuration
The following table lists the two configurations that have been tested:

| | Minimum configuration | Recommended configuration |
|----------|------------------------|----------------------------|
| **Operating system** | Windows 11 Professional | Windows 11 |
| **Processor (CPU)** | Intel Core i5-10210U (4 cores / 8 threads @ 1.6 GHz) | Intel Core i9-13900 (24 cores / 32 threads @ 2 GHz) |
| **Manufacturer** | Dell Inc. | Dell Inc. |
| **Model** | Latitude 5410 | Precision 3660 |
| **Graphics card (GPU)** | *CPU only* | NVIDIA RTX A2000 12 GB |
| **RAM** | ≥ 8 GB | ≥ 32 GB |
| **Free disk space** | ≥ 10 GB | ≥ 100 GB |
| **Python** | ≥ 3.10 | ≥ 3.11 |
| **PyTorch** | CPU version | CUDA 12.x version |

***Note:** SAM 2.1 can run without a GPU, but inference will be significantly slower.  
For batch or high-resolution image processing, an NVIDIA GPU (≥ 12 GB VRAM) is recommended.*

---

### Basic Installation on Windows

1. Install **Python 3.10 or 3.11** from [python.org/downloads](https://www.python.org/downloads/).  
   ➤ Check **“Add Python to PATH”** during installation.  
2. Install **Git** from [git-scm.com/downloads](https://git-scm.com/downloads).  
   ➤ Check **“Git Bash Here”** to add a context menu option.

---

## 2. Create a working folder and clone SAMourAI

Open **PowerShell** or **Command Prompt**, then run:

```bash
cd C:\Users\<user>\Documents
git clone https://github.com/cgalinier/SAMourAI.git

```

## 3. Clone the official SAM 2 repository
```bash
cd SAMourAI
git clone https://github.com/facebookresearch/segment-anything-2.git
```

## 4. Create & activate a virtual environment
```bash
python -m venv samourai_env
samourai_env\Scripts\activate
```

Once activated, the prompt will show the (samourai_env) prefix.

## 5. Install dependencies

From the `SAMourAI/` folder:
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
If you do not have an NVIDIA GPU, run instead:
```bash
pip install torch torchvision torchaudio
```

## 6. Download SAM 2 models (checkpoints)

Open Git Bash and run:
```bash
mkdir -p /c/Users/<USER>/Documents/SAMourAI/checkpoints
cd /c/Users/<USER>/Documents/SAMourAI/checkpoints
curl -O https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh
```
This script downloads the following model files:
- `sam2.1_hiera_tiny.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_large.pt`

## 7. SAM 2  configurations

Verify that the following folder exists:
`SAMourAI/segment-anything-2/sam2/configs/sam2.1/`

And that it contains:
- `sam2.1_hiera_t.yaml`
- `sam2.1_hiera_s.yaml`
- `sam2.1_hiera_b+.yaml`
- `sam2.1_hiera_l.yaml`

## 8. Launching SAMourAI
First run (from the virtual environment):
```bash
cd C:\Users\<user>\Documents\SAMourAI
samourai_env\Scripts\activate
pip install -r requirements.txt
```

**Execution**


**On Windows**
- CPU version: run `SAMourAI_lite.bat`
- GPU version: run `SAMourAI.bat`

**On Linux**
- CPU version: run
```bash
chmod +x Linux/SAMourAI_lite.sh
./Linux/SAMourAI_lite.sh
```
➜ `SAMourAI_lite.sh` is now an executable file (double-click)

- GPU version: run
```bash
chmod +x Linux/SAMourAI.sh
./Linux/SAMourAI.sh
```
➜ `SAMourAI.sh` is now an executable file (double-click)

➠ Segmentation masks are saved in the `masks/` folder.

### 9. Project structure
```bash
SAMourAI/
├── assets/                   # Resources (application icon)
│   └── icon.ico
├── build/                    
├── checkpoints/              # Pretrained SAM model files
│   ├── sam2.1_hiera_base_plus.pt
│   └── ...
├── gui/                      # Graphical interface
│   ├── __init__.py
│   ├── gui_cpu.py            # CPU version
│   └── gui_gpu.py            # GPU version
├── image_dir/                # Folder of images to segment
├── launchers/                # Launch scripts
│   ├── launcher_cpu.py
│   └── launcher_gpu.py
├── Linux/                    
│   ├── SAMourAI.sh           # GPU interface launcher (Linux version)
│   └── SAMourAI_lite.sh      # CPU interface launcher (Linux version)
├── masks/                    # Segmentation masks outputs
├── samourai_env/             # Virtual environment
├── segment-anything-2/       # Integrated SAM 2 repository
├── LICENSE
├── README.md
├── requirements.txt
├── SAMourAI.bat              # GPU launcher (Windows version)
└── SAMourAI_lite.bat         # CPU launcher (lite, Windows version)

```

## 10. Resources

- **SAMBIOTIC**  
  [https://github.com/jeremyfix/sambiotic](https://github.com/jeremyfix/sambiotic)  
- **SAM Documentation :**  
  [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)  
