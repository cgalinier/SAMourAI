
# LISEZMOI : Installation et utilisation de l’outil **SAMourAI** pour la segmentation d’images
*(Basé sur le dépôt [SAMBIOTIC](https://github.com/jeremyfix/sambiotic) de Jérémy Fix)*
Actualisé par le dépot [SAMourAI](https://github.com/cgalinier/SAMourAI.git)

> **Auteur :** Corentin GALINIER, Université de Lorraine 
> **Date de création :** 14/10/2025
---

## Objectif
  **SAMourAI** est un outil de segmentation semi-automatique d'images, utilisant le modèle [SAM2](https://github.com/facebookresearch/sam2) de *facebookresearch (Meta)*.

## 1. Pré-requis

### Configuration matérielle

| Élément | Configuration minimale | Configuration recommandée |
|----------|------------------------|----------------------------|
| **Système d’exploitation** | Microsoft Windows 11 Professionnel | Windows 11 |
| **Processeur (CPU)** | Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz, 2112 MHz, 4 cœur(s), 8 processeur(s) logique(s) | 13th Gen Intel(R) Core(TM) i9-13900, 2000 MHz, 24 cœur(s), 32 processeur(s) logique(s) |
| **Fabricant** | Dell Inc. | Dell Inc. |
| **Modèle**| Latitude 5410 | Precision 3660|
| **Carte graphique (GPU)** | ✖ *CPU uniquement* | NVIDIA RTX A2000 12GB |
| **Mémoire vive (RAM)** | ≥ 8 Go | ≥ 32 Go |
| **Espace disque disponible** | ≥ 10 Go | ≥ 100 Go |
| **Python** | ≥ 3.10 | ≥ 3.11 avec dépendances à jour |
| **PyTorch** | Version CPU | Dernière version stable compatible CUDA 12.x |

💡 *Remarque : SAM 2.1 peut fonctionner sans GPU, mais l’inférence est significativement plus lente. Pour un traitement d’images en série ou de haute résolution, un GPU NVIDIA avec ≥ 12 Go de VRAM est fortement recommandé.*

### Sous Windows
1. Installez **Python 3.10 ou 3.11** depuis [python.org/downloads](https://www.python.org/downloads/).
2. Pendant l’installation, cochez **“Add Python to PATH”**.
3. Installez **Git** depuis [git-scm.com](https://git-scm.com/downloads). Pendant l’installation, s'assurer que “Git Bash Here” est coché..

## 2. Créer un dossier de travail et cloner SAMourAI

Ouvrez un terminal (**Invite de commandes** ou **PowerShell**) puis exécutez :

```bash
cd C:\Users\<user>\Documents
git clone https://github.com/cgalinier/SAMourAI.git
```
## 3. Cloner le dépôt officiel SAM2 dans SAMourAI
```bash
cd SAMourAI
git clone https://github.com/facebookresearch/segment-anything-2.git
```

## 4. Créer et activer un environnement virtuel
```bash
python -m venv samourai_env
samourai_env\Scripts\activate
```
You should see (samourai_env) at the beginning of the line.

## 5. Installer les dépendances
Depuis le dossier SAMourAI, exécutez :
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
Si vous n’avez pas de GPU Nvidia, utilisez cette ligne à la place :
```bash
pip install torch torchvision torchaudio
```

## 6. Télécharger les modèles / chekpoints SAM2

Créez un dossier pour stocker les modèles et téléchargez-les automatiquement avec le script officiel :
Ouvrir **Git Bash**, puis copier-coller

```bash
mkdir -p /c/Users/<*USER*>/Documents/SAMourAI/checkpoints
cd /c/Users/<*USER*>/Documents/SAMourAI/checkpoints
curl -O https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh
```
Ce script télécharge tous les modèles SAM2 :
- sam2.1_hiera_tiny.pth
- sam2.1_hiera_small.pth
- sam2.1_hiera_base_plus.pth
- sam2.2_hiera_large.pth

## 7. Configurations SAM2

Assurez-vous que le dossier suivant existe : `SAMourAI/segment-anything-2/sam2/configs/sam2.1/` et qu’il contient les fichiers YAML nécessaires :
- sam2.1_hiera_t.yaml
- sam2.1_hiera_s.yaml
- sam2.1_hiera_b+.yaml
- - sam2.1_hiera_l.yaml

## 8. Lancer SAMourAI  
**Via l’environnement virtuel**
1er lancement :
```bash
cd C:\Users\<user>\Documents\SAMourAI\
samourai_env\Scripts\activate
pip install -r requirements.txt
```
**Exemple d’exécution**

Pour utiliser SAMourAI sur un laptop sans GPU, lancer "SAMourAI_lite" shortcut.
Pour utiliser SAMourAI sur un pc possédant un GPU, lancer "SAMourAI" shortcut.


Les résultats seront enregistrés dans le dossier `masks`.
---


🔑 Différences Clés
FonctionnalitéGPU VersionCPU VersionModèle par défautLargeTinyTaille max image2048px1024pxPrédiction pendant drag✅ Oui❌ NonThrottling❌ Non✅ 300msPréchargement❌ Non✅ OuiResamplingLANCZOSNEARESTChangement de modèle✅ Sans warning⚠️ Avec warning


![alt text](image.png)

```bash
SAMourAI/
├── gui_gpu.py          # Version GPU
├── gui_cpu.py          # Version CPU  
├── README.md                # Documentation
├── assets/
│   └── icon.ico            # Icône (optionnel)
└── sam2/                   # Dossier SAM2
    ├── configs/
    │   └── sam2.1/
    └── checkpoints/
        ├── sam2.1_hiera_tiny.pt
        ├── sam2.1_hiera_small.pt
        ├── sam2.1_hiera_base_plus.pt
        └── sam2.1_hiera_large.pt

```
## 9. Annexes

- **SAMBIOTIC**  
  [https://github.com/jeremyfix/sambiotic](https://github.com/jeremyfix/sambiotic)  
- **Documentation SAM :**  
  [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/sam2)  
