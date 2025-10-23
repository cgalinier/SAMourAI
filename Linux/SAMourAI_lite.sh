#!/bin/bash
clear
echo
echo "╔──────────────────────╗"
echo "│╔═╗╔═╗╔╦╗┌─┐┬ ┬┬─┐╔═╗╦│"
echo "│╚═╗╠═╣║║║│ ││ │├┬┘╠═╣║│"
echo "│╚═╝╩ ╩╩ ╩└─┘└─┘┴└─╩ ╩╩│"
echo "╚──────────────────────╝"
echo "     /lite version/"
echo "▬▬[════════════════════- "
echo
echo "[Initializing system...]"
echo

PROJECT_PATH="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$PROJECT_PATH/samourai_env/bin/activate"

if [ ! -f "$VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found: $VENV_PATH"
    read -p "Press Enter to exit..."
    exit 1
fi

# Activer l’environnement virtuel
source "$VENV_PATH"

IMAGE_DIR="$PROJECT_PATH/image_dir"
mkdir -p "$IMAGE_DIR"

echo "[Launching SAMourAI CPU Interface...]"
python "$PROJECT_PATH/launchers/launcher_cpu.py"

echo
echo "[Session ended]"
read -p "Press Enter to close..."