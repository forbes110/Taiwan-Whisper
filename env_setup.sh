conda create -n whisper python=3.10
conda activate whisper
pip install -e .
apt-get update && sudo apt-get install ffmpeg