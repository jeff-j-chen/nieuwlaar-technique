#!/bin/bash
# fetch_models.sh - download the five checkpoints into models/villa_ensemble/ and verify them.
# sha256 values for dense_native, ink_9um and Hecate are those of Nieuwlaar/pherc343-first-letters ink/models.tsv.
set -euo pipefail
cd "$(dirname "$0")"
D=models/villa_ensemble
PY=${VILLA_PY:-/opt/villa-venv/bin/python}
mkdir -p "$D"
hf download Nieuwlaar/ink9um-dense-native weights/dense_native_016000.safetensors training/configs/train_dense_native.json --local-dir "$D"
hf download scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth hybrid_3d2d-seed43/step-075000.pth --local-dir "$D"
hf download scrollprize/hecate hecate_9.6um.pth hecate.py --local-dir "$D"
hf download YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062 --local-dir "$D/v8in_loo_w062" --exclude "training/*"
(cd "$D" && sha256sum -c <<'EOF'
d3d95dc41dcb24ed76b84b5e12ca5dae25d7fb34edf1dfbb93e0186f91e6fc6d  weights/dense_native_016000.safetensors
5d8459030b36cd3980a03610c684365a7407d497173eff433efdb37f75840c86  training/configs/train_dense_native.json
e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab  hybrid_3d2d-seed42/step-075000.pth
2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f  hybrid_3d2d-seed43/step-075000.pth
809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d  hecate_9.6um.pth
c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59  hecate.py
EOF
)
# villa's flat inference needs a .pth that carries the training config (recipe from the dense_native model card)
"$PY" - "$D" <<'EOF'
import json, sys, torch
from safetensors.torch import load_file
d = sys.argv[1]
torch.save({"model": load_file(f"{d}/weights/dense_native_016000.safetensors"),
            "config": json.load(open(f"{d}/training/configs/train_dense_native.json")), "step": 16000},
           f"{d}/dense_native-016000.rebuilt.pth")
print("wrote", f"{d}/dense_native-016000.rebuilt.pth")
EOF
