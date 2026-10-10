# AGENT_INITIALIZATION

Instructions for an agent picking up this repo: how to get the surface volumes (zarrs), run the four-model ink
ensemble on them, and add new scrolls/segments. Read this whole file before running anything.

## 0. Ground rules

- **One GPU job at a time.** Never start a model run while another one (yours or the user's) is using the GPU; check
  `nvidia-smi` first and ask if it is busy. Queue jobs sequentially (see section 6).
- Two Python environments are involved:
  - your **normal Python** (>= 3.10, `requirements.txt`, `zarr<3`): runs the notebook, assembly and the ensemble;
  - the **villa env** (`/opt/villa-venv`, Python 3.14, torch 2.13, ScrollPrize/villa @ `0fb38c45`): only ever started
    by `utils/villa_ensemble.run_models()` as a subprocess (`python -P utils/villa_driver.py ...`). Never select it
    as the notebook kernel (the setup cell refuses to run in it).
- Everything below is run from the repo root.

## 1. One-time setup

```bash
pip install -r requirements.txt   # normal Python; provides the `hf` CLI (huggingface_hub)
bash setup_villa_env.sh           # villa env at /opt/villa-venv (needs a CUDA 13 driver; else TORCH_BACKEND=cu128)
bash fetch_models.sh              # all checkpoints -> models/villa_ensemble/, sha256-verified
```

`fetch_models.sh` downloads (all public, no HF login needed): dense_native step 16000 (and rebuilds
`dense_native-016000.rebuilt.pth`), ink_9um hybrid_3d2d seeds 42 and 43, Hecate 9.6 µm, and the v8-in
PHerc1447 LOO-w062 fine-tune. A different env location needs `VILLA_VENV=... VILLA_DIR=...` for the setup script and
`export VILLA_PY=<venv>/bin/python` for everything after.

## 2. Getting the zarrs

All inputs are 28-layer surface volumes `ves_zarrs2/<id>.zarr` (shape `(28, H, W)`, layer k sampled at k − 13.5
voxels along the mesh normal in the **native** voxel grid of the source CT) plus a valid-surface mask
`masks/<id>.png`. They are not stored in git; they are rendered from the tifxyz meshes in `tifxyz/`:

```bash
python assemble_test_segments.py                       # every segment in FRAGMENTS
python assemble_test_segments.py --only 20260928000008 # one segment (repeat --only for several)
python assemble_test_segments.py --force --only <id>   # re-render (keeps the downloaded raw chunks)
```

- The script streams the raw 128³ uint8 CT chunks the mesh touches from
  `https://vesuvius-challenge-open-data.s3.amazonaws.com/<SCROLL>/volumes/<volume>.zarr/0` (public, no credentials),
  caching them in `_ves_tmp/render_<id>/` (2–4 GB per segment; delete afterwards to save disk).
- It stops when free disk drops below `--min-free-gb` (default 10).
- The mask is written **last**; a zarr whose mask is missing or older than its `.zarray` was interrupted
  (`--force --only <id>`).
- The notebook calls the assembler automatically for any missing segment, so explicit assembly is optional.
- Sanity segment seg46527 (PHerc0814, known ink, id `20260226000000`) is not mesh-rendered:
  `python fetch_sanity_segment.py` downloads its published surface volume straight into
  `ves_zarrs_villa/20260226000000.zarr/0` (the notebook does this itself).

`ves_zarrs_villa/<id>.zarr/0` is the models' input (uint8 group copy of the render, made by
`ve.ensure_villa_input`, cached; full-range uint16 sources are windowed to 1..255).

## 3. Running the ensemble

Notebook: open `villa_ensemble_inference.ipynb` with the **normal** Python kernel, run cell 2 (setup), then the
sanity cell, then any segment cell. Settings in cell 2:

| setting | meaning |
|---|---|
| `SLOT3` | third model: `"v8in-1447"` (PHerc1447 fine-tune, by far the slowest) or `"ink9-42"` (ink_9um seed 42, fast). dense_native, Hecate and seed 43 are always used |
| `ve.V8IN_STRIDE` | v8-in tile stride: `None` = 21 (full quality), `32` = 2.3x faster (r = 0.940 vs 0.954 for stride 21 against a dense stride-8 map) |
| `DEPTHS`, `WRITE_DEPTH` | depth offset(s) d. 28-layer renders allow d ∈ [−5, 5] for dense_native/ink_9um/Hecate, only [−2, 2] for v8-in; `render_depths(name)` sweeps −5..+5 and leaves a model out where it cannot reach |
| `DIRECTION` | `"forward"` = layer order that reads the text side of these renders (validated on seg46527: ink 0.67 vs background 0.14; reverse shows nothing) |

Headless (same code as the notebook; prefer this for long runs):

```bash
nohup python -u - > output/run_<name>.log 2>&1 <<'EOF' &
import json
nb = json.load(open("villa_ensemble_inference.ipynb"))
ns = {}
exec("".join(nb["cells"][1]["source"]), ns)   # setup cell
ns["ve"].set_slot3("ink9-42")                 # optional: fast third model
ns["ve"].V8IN_STRIDE = 32                     # optional: faster v8-in
for name in ["PHerc0813_z13088_w040_abf"]:   # keys of FRAGMENTS in cell 2
    ns["render"](name)                        # or ns["render_depths"](name)
EOF
```

Per segment `render(name)`:
1. `ensure_assembled` → `ves_zarrs2/<id>.zarr`, `masks/<id>.png`
2. `ve.ensure_villa_input` → `ves_zarrs_villa/<id>.zarr/0`
3. `ve.run_models` → raw maps in `output/villa_ensemble_maps/<id>/` (only missing maps are computed; driver log in
   `driver.log` there). Map names encode model, depth and (for v8-in at a non-default stride) `st<stride>`, so
   different settings never overwrite each other.
4. figures (saved, not shown) in `output/test_visualizations/villa_ensemble/`:
   - `<name>_4models_d+0.png`: 2x2 grid of the per-model maps (each stretched p1–p99.5), full resolution
   - `<name>_ensemble_d+0.png`: mean of the per-model z-scores (stretched p1–p99.5), with the composite view
     (max of raw layers 10–17, red 1 cm bar) below it
   - `render_depths` writes the same pair per depth to `<name>_depths/`

Ensemble image without titles/composite, on the exact zarr/mask grid (e.g. to compare with a label):

```python
_, ensemble, _ = ns["ve"].ensemble(scroll_id, ns["DEPTHS"], ns["DIRECTION"], ns["WRITE_DEPTH"])  # float 0..1, NaN = invalid
```

## 4. Adding a new segment (new scroll or new mesh)

1. **Mesh**: put the tifxyz folder (`x.tif`, `y.tif`, `z.tif`, `meta.json`; one grid step = 20 voxels) in
   `tifxyz/<mesh_name>/`. Its coordinates are level-0 voxels of exactly one CT volume.
2. **Volume**: find that volume in the open-data bucket and read its shape:
   ```bash
   curl -s "https://vesuvius-challenge-open-data.s3.amazonaws.com/?list-type=2&prefix=<SCROLL>/volumes/&delimiter=/"
   curl -s https://vesuvius-challenge-open-data.s3.amazonaws.com/<SCROLL>/volumes/<volume>.zarr/0/.zarray
   ```
   It must be uint8 with 128³ chunks, and the mesh bbox in `meta.json` must lie inside `shape` (z, y, x). The voxel
   size must appear in the volume name as `-<um>um-` (e.g. `-8.640um-`); `ve.native_um` parses it from there.
3. **Register** it in `assemble_test_segments.py` → `FRAGMENTS`:
   `("<id>", "<mesh_name>", f"{BUCKET}/<SCROLL>/volumes/<volume>.zarr/0", "<z>,<y>,<x>")`.
   Ids are unique 14-digit strings; for meshes without a timestamp use the next free synthetic id
   (`202609280000NN`; `...0008` is taken).
4. **Notebook**: add `"<name>": <id>` to `FRAGMENTS` in cell 2 and a cell `render("<name>")`.
5. **Assemble**: `python assemble_test_segments.py --only <id>`.
6. **Orientation check** (do not skip): run `render("<name>")` once with the fast set (`SLOT3 = "ink9-42"`; this
   costs about 1–2 min per 10 Mpx). If the ensemble shows no letters or ink-like structure where the composite view
   shows papyrus, the mesh normal is probably flipped: add `"<id>"` to `depth_reversed.json`, then
   ```bash
   python assemble_test_segments.py --force --only <id>
   rm -rf ves_zarrs_villa/<id>.zarr output/villa_ensemble_maps/<id>
   ```
   and render again (the villa input copy and the maps are cached, so they must be deleted after a re-render).
   Keep `DIRECTION = "forward"` for all segments; orientation is fixed per segment through `depth_reversed.json`.
7. Final run with the settings the user wants (usually `SLOT3 = "v8in-1447"`).

A segment that is already a published surface volume (no mesh) can be used like the sanity segment: copy its
level-0 uint8 `(28, H, W)` array to `ves_zarrs_villa/<id>.zarr/0` (see `fetch_sanity_segment.py`), add the id to
`ve.EXTRA_NATIVE_UM` with its voxel size, and special-case it in `ensure_assembled`.

## 5. Expected run times (RTX PRO 4500 Blackwell, 32 GB)

Measured on `PHerc0813_z13088_w040_abf` (2301 x 4101 px), one depth:

| model | time |
|---|---|
| dense_native | 15 s |
| Hecate (bf16) | 66 s |
| ink_9um seed 42 / 43 | 14 s each |
| v8-in, stride 21 | 444 s |
| v8-in, stride 32 | ~195 s (estimated from the 2.3x stride benchmark) |

Time scales with the valid surface area; over many segments the whole ensemble with v8-in at stride 32 averaged
**0.76 min per million valid pixels**. Each extra depth costs the same again (v8-in only for |d| ≤ 2). v8-in uses
~17 GB of GPU memory. Hecate and v8-in are compute-bound: larger batches do not help (measured).

## 6. Long runs and queueing

- Use `render_all()` (last notebook cell) or a headless script. Figures are always redrawn, but every finished map is
  reused, so a restart only loses the map that was in progress. `render_all` logs failures and continues.
- To run several jobs in sequence, put them in one shell script and start it with `nohup`; never start a second GPU
  job while one is running. Stopping a run: kill the parent script, the python process and any `villa_driver.py`.

## 7. Facts worth knowing

- Ensemble (reference `ensemble_maps.py`): villa uint8 maps decode as clip((v − 64)/128, 0, 1), Hecate as v/255
  (resized to the render), v8-in as uint16/65535 with 0 = not predicted (treated as missing). Each model is
  stretched to its own p1–p99.5 over valid pixels (CT > 0 in every filled layer); the ensemble is the mean of the
  per-model z-scores over the models that predicted a pixel, stretched p1–p99.5.
- v8-in was fine-tuned on PHerc1447 windings w058 and w060 (w062 left out); it is not an independent judge there.
- Hecate runs in bf16 (`ve.HECATE_PRECISION`): 1.65x faster than fp32, mean difference 0.13/255.
- `utils/villa_driver.py` must be started with `python -P` (done by `run_models`).
- Reference method: https://github.com/Nieuwlaar/pherc343-first-letters (villa commit `0fb38c45`).
