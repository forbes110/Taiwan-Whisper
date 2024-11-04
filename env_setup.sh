conda create -n whisper python=3.10
conda activate whisper
pip install -e .
apt-get update && apt-get install ffmpeg
cd ..
git clone https://github.com/SYSTRAN/faster-whisper.git
cd faster-whisper
pip install -e .
cd ..