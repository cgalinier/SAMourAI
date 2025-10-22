# SAMourAI — Guide d’installation et d’utilisation  
*(Basé sur le dépôt [SAMBIOTIC](https://github.com/jeremyfix/sambiotic) de Jérémy Fix)*
> **Auteur :** Corentin GALINIER — Université de Lorraine  
> **Date de création :** 22/10/2025  

---

**SAMourAI** est un outil de **segmentation semi-automatique d’images**, reposant sur le modèle [**SAM 2**](https://github.com/facebookresearch/sam2) développé par *Meta (Facebook Research)*.  
Il offre une **interface graphique** permettant d’effectuer la segmentation d’images, en mode **CPU** ou **GPU**.

---
![UI](assets/ui.png)

---
## 1. Pré-requis

### Configuration matérielle
Ce tableau considère les deux configurations testées suivantes : 

| | Configuration minimale | Configuration recommandée |
|----------|------------------------|----------------------------|
| **Système d’exploitation** | Windows 11 Professionnel | Windows 11 |
| **Processeur (CPU)** | Intel Core i5-10210U (4 cœurs / 8 threads @ 1.6 GHz) | Intel Core i9-13900 (24 cœurs / 32 threads @ 2 GHz) |
| **Fabricant** | Dell Inc. | Dell Inc. |
| **Modèle** | Latitude 5410 | Precision 3660 |
| **Carte graphique (GPU)** | *CPU uniquement* | NVIDIA RTX A2000 12 Go |
| **Mémoire vive (RAM)** | ≥ 8 Go | ≥ 32 Go |
| **Espace disque disponible** | ≥ 10 Go | ≥ 100 Go |
| **Python** | ≥ 3.10 | ≥ 3.11 |
| **PyTorch** | Version CPU | Version CUDA 12.x |

***Remarque** : SAM 2.1 peut fonctionner sans GPU, mais l’inférence est significativement plus lente.  
Pour un traitement d’images en série ou haute résolution, un GPU NVIDIA (≥ 12 Go VRAM) est recommandé.*

---

### Installation de base sous Windows

1. Installez **Python 3.10 ou 3.11** depuis [python.org/downloads](https://www.python.org/downloads/).  
   ➤ Cochez **“Add Python to PATH”** pendant l’installation.
2. Installez **Git** depuis [git-scm.com/downloads](https://git-scm.com/downloads).  
   ➤ Cochez **“Git Bash Here”** pour ajouter un menu contextuel.

---

## 2. Créer un dossier de travail et cloner SAMourAI

Ouvrez **PowerShell** ou **Invite de commandes**, puis exécutez :

```bash
cd C:\Users\<user>\Documents
git clone https://github.com/cgalinier/SAMourAI.git
```

## 3. Cloner le dépôt officiel SAM 2
```bash
cd SAMourAI
git clone https://github.com/facebookresearch/segment-anything-2.git
```

## 4. Créer et activer un environnement virtuel

```bash
python -m venv samourai_env
samourai_env\Scripts\activate
```

Une fois activé, le préfixe (samourai_env) apparaît au début de la ligne de commande.

## 5. Installer les dépendances

Depuis le dossier `SAMourAI/`:
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
Si vous n’avez pas de GPU NVIDIA, exécutez plutôt :
```bash
pip install torch torchvision torchaudio
```

## 6. Télécharger les modèles (checkpoints) SAM 2

Ouvrez Git Bash, puis copiez-collez :
```bash
mkdir -p /c/Users/<USER>/Documents/SAMourAI/checkpoints
cd /c/Users/<USER>/Documents/SAMourAI/checkpoints
curl -O https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh
```
Ce script télécharge les modèles :
- `sam2.1_hiera_tiny.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_large.pt`

## 7. Configurations SAM 2

Vérifiez que le dossier suivant existe :
`SAMourAI/segment-anything-2/sam2/configs/sam2.1/`

Et qu’il contient les fichiers :
- `sam2.1_hiera_t.yaml`
- `sam2.1_hiera_s.yaml`
- `sam2.1_hiera_b+.yaml`
- `sam2.1_hiera_l.yaml`

## 8. Lancer SAMourAI
1ère exécution (depuis l’environnement virtuel)
```bash
cd C:\Users\<user>\Documents\SAMourAI
samourai_env\Scripts\activate
pip install -r requirements.txt
```

**Exécution**


**Sur Windows**
- CPU version:lancer `SAMourAI_lite.bat`
- GPU version: lancer `SAMourAI.bat`

**Sur Linux**
- CPU version: à la 1ère utilisation, lancer
```bash
chmod +x Linux/SAMourAI_lite.sh
./Linux/SAMourAI_lite.sh
```
➜ `SAMourAI_lite.sh` est à présent un fichier exécutable (double-clic)

- GPU version: à la 1ère utilisation, lancer
```bash
chmod +x Linux/SAMourAI.sh
./Linux/SAMourAI.sh
```
➜ `SAMourAI.sh` est à présent un fichier exécutable (double-clic)

➠Les masques sont enregistrés dans le dossier `masks/`.

### 9. Structure du projet
```bash
SAMourAI/
├── assets/                   # Ressources (icône de l’application)
│   └── icon.ico
├── build/                    
├── checkpoints/              # Modèles pré-entraînés SAM
│   ├── sam2.1_hiera_base_plus.pt
│   └── ...
├── gui/                      # Interface graphique
│   ├── __init__.py
│   ├── gui_cpu.py            # Version CPU
│   └── gui_gpu.py            # Version GPU
├── image_dir/                # Dossier d’images à segmenter
├── launchers/                # Scripts de lancement
│   ├── launcher_cpu.py
│   └── launcher_gpu.py
├── masks/                    # Masques de segmentation
├── samourai_env/             # Environnement virtuel
├── segment-anything-2/       # Dépôt SAM 2 intégré
├── LICENSE
├── README.md
├── requirements.txt
├── SAMourAI.bat              # Lancement GPU
└── SAMourAI_lite.bat         # Lancement CPU (lite)
```

## 10. Annexes

- **SAMBIOTIC**  
  [https://github.com/jeremyfix/sambiotic](https://github.com/jeremyfix/sambiotic)  
- **Documentation SAM :**  
  [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)  
