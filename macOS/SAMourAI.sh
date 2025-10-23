clear
echo
echo "╔──────────────────────╗"
echo "│╔═╗╔═╗╔╦╗┌─┐┬ ┬┬─┐╔═╗╦│"
echo "│╚═╗╠═╣║║║│ ││ │├┬┘╠═╣║│"
echo "│╚═╝╩ ╩╩ ╩└─┘└─┘┴└─╩ ╩╩│"
echo "╚──────────────────────╝"
echo "  /macOS GPU version/"
echo "▬▬[════════════════════- "
echo
echo "[Initializing SAMourAI system on macOS...]"
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

echo
echo "[Checking PyTorch Metal (MPS) support...]"
python3 - <<'EOF'
import torch
if torch.backends.mps.is_available():
    print("Apple GPU detected — using Metal (MPS) backend.")
else:
    print("Running on CPU — MPS not available.")
EOF

IMAGE_DIR="$PROJECT_PATH/image_dir"
mkdir -p "$IMAGE_DIR"

echo
echo "[Launching SAMourAI GPU Interface...]"
echo "Project path: $PROJECT_PATH"
echo
python3 "$PROJECT_PATH/launchers/launcher_gpu.py"

echo
echo "[Session ended]"
read -p "Press Enter to close..."