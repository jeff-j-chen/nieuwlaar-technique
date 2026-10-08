#!/bin/bash
# setup_villa_env.sh - isolated inference env: Python 3.14 + torch 2.13 + ScrollPrize/villa @ 0fb38c45 (the commit the
# PHerc0343 First Letters maps were made with). The notebook itself runs in your normal Python; it calls this env.
#
#   bash setup_villa_env.sh              # VILLA_VENV=/opt/villa-venv VILLA_DIR=/opt/villa TORCH_BACKEND=cu130
set -euo pipefail
VENV=${VILLA_VENV:-/opt/villa-venv}
VILLA=${VILLA_DIR:-/opt/villa}
BACKEND=${TORCH_BACKEND:-cu130}
COMMIT=0fb38c45b488e192566ea83ae36369cb6db92652

command -v uv >/dev/null || python3 -m pip install -q uv
if [[ ! -d $VILLA/.git ]]; then
  mkdir -p "$VILLA" && git -C "$VILLA" init -q
  git -C "$VILLA" remote add origin https://github.com/ScrollPrize/villa.git
  git -C "$VILLA" sparse-checkout set vesuvius
fi
git -C "$VILLA" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git -C "$VILLA" checkout -q FETCH_HEAD

uv python install 3.14
[[ -x $VENV/bin/python ]] || uv venv -q --python 3.14 "$VENV"
PY=$VENV/bin/python
uv pip install -q --python "$PY" --index-url "https://download.pytorch.org/whl/$BACKEND" torch==2.13.0
uv pip install -q --python "$PY" "numpy>=2.0.2" pillow pynrrd requests pyyaml fsspec "numcodecs>=0.15.1,<0.17" \
  "zarr>=2.18.7,<4" s3fs cachetools edt tifffile imagecodecs scipy opencv-python-headless safetensors tqdm einops \
  timm dynamic-network-architectures huggingface-hub nest-asyncio
uv pip install -q --python "$PY" --no-deps -e "$VILLA/vesuvius"
"$PY" -c "import torch, vesuvius.ink_detection.inference.infer; x = torch.ones(8, device='cuda'); print('villa env ok:', torch.__version__, torch.cuda.get_device_name(0), float(x.sum()))"
