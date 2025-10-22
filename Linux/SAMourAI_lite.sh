#!/bin/bash
clear
echo
echo "╔──────────────────────╗"
echo "│╔═╗╔═╗╔╦╗┌─┐┬ ┬┬─┐╔═╗╦│"
echo "│╚═╗╠═╣║║║│ ││ │├┬┘╠═╣║│"
echo "│╚═╝╩ ╩╩ ╩└─┘└─┘┴└─╩ ╩╩│"
echo "╚──────────────────────╝"
echo "      /GPU version/"
echo "▬▬[════════════════════- "
echo
echo "[Initializing system...]"
echo

# ───────────────────────────────
# Détection du chemin du projet
# ───────────────────────────────
PROJECT_PATH="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$PROJECT_PATH/samourai_env/bin/activate"

# ───────────────────────────────
# Vérification de l'environnement virtuel
# ───────────────────────────────
if [ ! -f "$VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found: $VENV_PATH"
    read -p "Press Enter to exit..."
    exit 1
fi

# ───────────────────────────────
# Activation de l'environnement virtuel
# ───────────────────────────────
source "$VENV_PATH"

# ───────────────────────────────
# Vérification / création du dossier image_dir
# ───────────────────────────────
IMAGE_DIR="$PROJECT_PATH/image_dir"
mkdir -p "$IMAGE_DIR"

# ───────────────────────────────
# Lancement de l'interface GPU
# ───────────────────────────────
echo "[Launching SAMourAI GPU Interface...]"
python "$PROJECT_PATH/launchers/launcher_gpu.py"

echo
echo "[Session ended]"
read -p "Press Enter to close..."
