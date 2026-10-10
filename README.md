# nieuwlaar-technique

Per-patch four-model ink ensemble, following the method of
[Nieuwlaar/pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters), applied to our test segments.

| model | source |
|---|---|
| dense_native (step 16k) | [Nieuwlaar/ink9um-dense-native](https://huggingface.co/Nieuwlaar/ink9um-dense-native) |
| ink_9um hybrid_3d2d seed 43 | [scrollprize/ink_9um](https://huggingface.co/scrollprize/ink_9um) |
| Hecate 9.6 µm (run in bf16) | [scrollprize/hecate](https://huggingface.co/scrollprize/hecate) |
| third model (`SLOT3`): v8-in PHerc1447 LOO-w062 (default, slowest) **or** ink_9um hybrid_3d2d seed 42 (fast) | [YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062](https://huggingface.co/YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062), [scrollprize/ink_9um](https://huggingface.co/scrollprize/ink_9um) |

Each model's map is contrast-stretched (p1–p99.5); the ensemble is the mean of the per-model z-scores
(`ensemble_maps.py` in the reference repo).

**Agents: start with [AGENT_INITIALIZATION.md](AGENT_INITIALIZATION.md)** (exact setup, getting the zarrs, adding new
segments, run times).

## Setup

Requires Linux, an NVIDIA GPU with a CUDA 13 driver (≥ 580; for 12.8 use `TORCH_BACKEND=cu128`), `git` and `curl`.

```bash
pip install -r requirements.txt      # notebook + assembly env (your normal Python)
bash setup_villa_env.sh              # inference env: /opt/villa-venv (Python 3.14, torch 2.13, villa @ 0fb38c45)
bash fetch_models.sh                 # checkpoints -> models/villa_ensemble/ (sha256-checked)
```

`VILLA_VENV`/`VILLA_DIR` change the env/checkout location; then set `VILLA_PY=<venv>/bin/python` for the notebook.

## Run

Open `villa_ensemble_inference.ipynb` from the repo root with your **normal** Python kernel (not the villa venv),
run the setup cell, then the sanity cell (seg46527, PHerc0814, known ink), then any fragment cell. Each cell:

1. renders the segment from its mesh in `tifxyz/` if needed (`assemble_test_segments.py`: streams raw CT chunks
   from the open-data bucket, ~2–4 GB cached per segment in `_ves_tmp/`) → `ves_zarrs2/<id>.zarr`, `masks/<id>.png`;
   the sanity segment is downloaded directly by `fetch_sanity_segment.py`
2. writes a uint8 copy in the models' input format → `ves_zarrs_villa/<id>.zarr/0`
3. runs the four models (progress bar with ETA) → raw maps in `output/villa_ensemble_maps/<id>/` (cached)
4. saves, not displayed, to `output/test_visualizations/villa_ensemble/`:
   `<name>_4models_d+0.png` (2×2, every panel at native resolution) and `<name>_ensemble_d+0.png` (ensemble with
   the composite view, max of raw layers 10–17 with a 1 cm bar, below it)

Settings in the setup cell: `SLOT3` (`"v8in-1447"` or `"ink9-42"`), `ve.V8IN_STRIDE` (`None` = 21; `32` is 2.3× faster
for a small fidelity loss), `DEPTHS` (depth offsets d, −5..5 for most models; v8-in −2..2), `DIRECTION`
(`forward` = reads the text side of these renders; verified on seg46527), `COMPILE`. `render_depths(name)` writes a
d = −5..+5 sweep to `<name>_depths/`.

To render all segments up front: `python assemble_test_segments.py` (or `--only <id>`).

## Files

| file | role |
|---|---|
| `AGENT_INITIALIZATION.md` | step-by-step guide for agents: setup, zarrs, running, new segments, timings |
| `villa_ensemble_inference.ipynb` | one cell per segment |
| `assemble_test_segments.py` | mesh → 28-layer surface volume (layer k at k − 13.5 voxels along the normal, native voxel grid) |
| `depth_reversed.json` | segments whose mesh normal is flipped; rendered in reversed layer order |
| `fetch_sanity_segment.py` | downloads seg46527's published surface volume |
| `utils/villa_ensemble.py` | input conversion, driver launch + progress, ensemble, figures |
| `utils/villa_driver.py` | runs inside the villa env; loads each model once, writes the raw maps |
| `setup_villa_env.sh`, `fetch_models.sh` | environment and checkpoints |
