clear
echo
echo "╔──────────────────────╗"
echo "│╔═╗╔═╗╔╦╗┌─┐┬ ┬┬─┐╔═╗╦│"
echo "│╚═╗╠═╣║║║│ ││ │├┬┘╠═╣║│"
echo "│╚═╝╩ ╩╩ ╩└─┘└─┘┴└─╩ ╩╩│"
echo "╚──────────────────────╝"
echo "  /macOS CPU version/"
echo "▬▬[════════════════════- "
echo
echo "[Initializing SAMourAI system on macOS CPU...]"
echo

PROJECT_PATH="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$PROJECT_PATH/samourai_env/bin/activate"

if [ ! -f "$VENV_PATH" ]; then
    echo "[ERROR] Virtual environment not found:"
    echo "    $VENV_PATH"
    echo
    echo "To create it, run:"
    echo "    cd \"$PROJECT_PATH\""
    echo "    python3 -m venv samourai_env"
    echo "    source samourai_env/bin/activate"
    echo "    pip install -r requirements.txt"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[Activating Python virtual environment...]"
source "$VENV_PATH"

IMAGE_DIR="$PROJECT_PATH/image_dir"
mkdir -p "$IMAGE_DIR"

echo
echo "[Launching SAMourAI CPU Interface...]"
echo "Project path: $PROJECT_PATH"
echo
python3 "$PROJECT_PATH/launchers/launcher_cpu.py"

echo
echo "[Session ended]"
read -p "Press Enter to close..."
