# load conda config
source /root/miniconda3/etc/profile.d/conda.sh

# create conda env
conda create -n whisper python=3.10 -y
conda activate whisper
pip install -e .
apt-get update && apt-get install -y ffmpeg

# setup faster-whisper
cd ..
git clone https://github.com/SYSTRAN/faster-whisper.git
cd faster-whisper
pip install -e .
cd ..
