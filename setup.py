import os
from setuptools import setup, find_packages

# Core dependencies
_deps = [
    "torch>=1.10",
    "transformers>=4.35.1",
    "datasets[audio]>=2.14.7",
    "accelerate>=0.24.1",
    "jiwer",
    "evaluate>=0.4.1",
    "wandb",
    "tensorboard",
    "nltk",
    "huggingface_hub>=0.13",
]

# Optional development dependencies
_extras_dev_deps = [
    "ruff==0.1.5",
]

# Additional pip packages
_additional_deps = [
    "av>=11",
    "absl-py==2.1.0",
    "aiohappyeyeballs==2.4.3",
    "aiohttp==3.10.10",
    "aiosignal==1.3.1",
    "alembic==1.13.3",
    "antlr4-python3-runtime==4.9.3",
    "asteroid-filterbanks==0.4.0",
    "attrs==24.2.0",
    "audioread==3.0.1",
    "certifi==2024.8.30",
    "cffi==1.17.1",
    "charset-normalizer==3.4.0",
    "click==8.1.7",
    "coloredlogs==15.0.1",
    "colorlog==6.8.2",
    "contourpy==1.3.0",
    "ctranslate2==4.4.0",
    "cycler==0.12.1",
    "decorator==5.1.1",
    "dill==0.3.8",
    "distance==0.1.3",
    "docker-pycreds==0.4.0",
    "docopt==0.6.2",
    "edit-distance==1.0.6",
    "editdistance==0.8.1",
    "einops==0.8.0",
    "filelock==3.16.1",
    "flatbuffers==24.3.25",
    "fonttools==4.54.1",
    "frozenlist==1.4.1",
    "fsspec==2024.6.1",
    "g2p-en==2.1.0",
    "gitdb==4.0.11",
    "gitpython==3.1.43",
    "greenlet==3.1.1",
    "grpcio==1.67.0",
    "humanfriendly==10.0",
    "humanize==4.11.0",
    "hyperpyyaml==1.2.2",
    "idna==3.10",
    "inflect==7.4.0",
    "jinja2==3.1.4",
    "jiwer==3.0.4",
    "joblib==1.4.2",
    "julius==0.2.7",
    "kiwisolver==1.4.7",
    "lazy-loader==0.4",
    "librosa==0.10.2.post1",
    "lightning==2.4.0",
    "lightning-utilities==0.11.8",
    "llvmlite==0.43.0",
    "mako==1.3.5",
    "markdown==3.7",
    "markdown-it-py==3.0.0",
    "markupsafe==3.0.2",
    "matplotlib==3.9.2",
    "mdurl==0.1.2",
    "more-itertools==10.5.0",
    "mpmath==1.3.0",
    "msgpack==1.1.0",
    "multidict==6.1.0",
    "multiprocess==0.70.16",
    "networkx==3.4.1",
    "numba==0.60.0",
    "numpy==1.26.4",
    "omegaconf==2.3.0",
    "onnxruntime==1.19.2",
    "openai-whisper==20240930",
    "opencc==1.1.9",
    "optuna==4.0.0",
    "packaging==24.1",
    "pandas==2.2.3",
    "pillow==11.0.0",
    "platformdirs==4.3.6",
    "pooch==1.8.2",
    "primepy==1.3",
    "propcache==0.2.0",
    "protobuf==5.28.2",
    "psutil==6.1.0",
    "pyannote-audio==3.3.2",
    "pyannote-core==5.0.0",
    "pyannote-database==5.1.0",
    "pyannote-metrics==3.2.1",
    "pyannote-pipeline==3.0.1",
    "pyarrow==17.0.0",
    "pycparser==2.22",
    "pydub==0.25.1",
    "pygments==2.18.0",
    "pyparsing==3.2.0",
    "pypinyin==0.53.0",
    "python-dateutil==2.9.0.post0",
    "pytorch-lightning==2.4.0",
    "pytorch-metric-learning==2.6.0",
    "pytz==2024.2",
    "pyyaml==6.0.2",
    "rapidfuzz==3.10.0",
    "regex==2024.9.11",
    "requests==2.32.3",
    "rich==13.9.2",
    "ruamel-yaml==0.18.6",
    "ruamel-yaml-clib==0.2.8",
    "safetensors==0.4.5",
    "scikit-learn==1.5.2",
    "scipy==1.14.1",
    "semver==3.0.2",
    "sentencepiece==0.2.0",
    "sentry-sdk==2.17.0",
    "setproctitle==1.3.3",
    "shellingham==1.5.4",
    "six==1.16.0",
    "smmap==5.0.1",
    "sortedcontainers==2.4.0",
    "sounddevice==0.5.1",
    "soundfile==0.12.1",
    "soxr==0.5.0.post1",
    "speechbrain==1.0.1",
    "sqlalchemy==2.0.36",
    "sympy==1.13.1",
    "tabulate==0.9.0",
    "tensorboard-data-server==0.7.2",
    "tensorboardx==2.6.2.2",
    "threadpoolctl==3.5.0",
    "tiktoken==0.8.0",
    "tokenizers==0.20.1",
    "torch-audiomentations==0.11.1",
    "torch-pitch-shift==1.2.5",
    "torchaudio==2.5.0",
    "torchmetrics==1.5.0",
    "tqdm==4.66.5",
    "triton==3.1.0",
    "typeguard==4.3.0",
    "typer==0.12.5",
    "typing-extensions==4.12.2",
    "tzdata==2024.2",
    "urllib3==2.2.3",
    "werkzeug==3.0.4",
    "xxhash==3.5.0",
    "yarl==1.15.5",
]

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name="TW-Whisper",
    version="0.1.0",
    description="A Python package for speech processing.",
    packages=find_packages(),
    install_requires=_deps + _additional_deps,
    extras_require={"dev": _extras_dev_deps},
)