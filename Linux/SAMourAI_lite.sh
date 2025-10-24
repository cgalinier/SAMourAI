#!/bin/bash
clear
echo
echo "╔──────────────────────╗"
echo "│╔═╗╔═╗╔╦╗┌─┐┬ ┬┬─┐╔═╗╦│"
echo "│╚═╗╠═╣║║║│ ││ │├┬┘╠═╣║│"
echo "│╚═╝╩ ╩╩ ╩└─┘└─┘┴└─╩ ╩╩│"
echo "╚──────────────────────╝"
echo "     /Lite version/"
echo "▬▬[════════════════════- "
echo
echo "[Initializing system...]"
echo

# -------------------------------------------------------------------
# Détection automatique du dossier racine du projet SAMourAI
# -------------------------------------------------------------------

SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)"

# Si le script est dans Linux/, remonte d’un dossier
if [[ "$(basename "$SCRIPT_PATH")" == "Linux" ]]; then
    PROJECT_PATH="$(cd "$SCRIPT_PATH/.." && pwd)"
else
    PROJECT_PATH="$SCRIPT_PATH"
fi

VENV_PATH="$PROJECT_PATH/samourai_env/bin/activate"

# Vérifie que l’environnement virtuel existe
if [ ! -f "$VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found: $VENV_PATH"
    read -p "Press Enter to exit..."
    exit 1
fi

# Active l’environnement virtuel
source "$VENV_PATH"

# Crée le dossier image_dir s’il n’existe pas
IMAGE_DIR="$PROJECT_PATH/image_dir"
mkdir -p "$IMAGE_DIR"

echo "[Launching SAMourAI Lite Interface...]"
python "$PROJECT_PATH/launchers/launcher_cpu.py"

echo
echo "[Session ended]"
read -p "Press Enter to close..."
