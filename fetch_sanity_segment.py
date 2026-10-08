#!/usr/bin/env python3
"""fetch_sanity_segment.py -- download the sanity-check segment seg46527 (PHerc0814, known ink) into
ves_zarrs_villa/20260226000000.zarr/0.

Level 0 of the published 28-layer surface volume is already in the models' input format (uint8, Z,Y,X, uncompressed
(28,128,128) chunks), so its chunks are copied verbatim; missing chunks are all-zero air.
"""
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0814/segments/20260226000000-46527_2um_try2/"
       "surface-volumes/9.362um-1.2m-113keV-volume-20250804134230.zarr/0")
REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.getenv("VILLA_ZARR_PATH", os.path.join(REPO, "ves_zarrs_villa")), "20260226000000.zarr")


def fetch(key):
    path = os.path.join(OUT, "0", *key.split("/"))
    if os.path.exists(path):
        return
    try:
        data = urllib.request.urlopen(f"{URL}/{key}", timeout=120).read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return
        raise
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".partial", "wb") as handle:
        handle.write(data)
    os.replace(path + ".partial", path)


def main():
    attrs_path = os.path.join(OUT, ".zattrs")
    if os.path.isfile(attrs_path) and json.load(open(attrs_path)).get("complete"):
        print(f"present: {OUT}")
        return
    meta = json.loads(urllib.request.urlopen(f"{URL}/.zarray", timeout=60).read())
    os.makedirs(os.path.join(OUT, "0"), exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(OUT, ".zgroup"), "w"))
    json.dump(meta, open(os.path.join(OUT, "0", ".zarray"), "w"), indent=2)
    grid = [-(-s // c) for s, c in zip(meta["shape"], meta["chunks"])]
    keys = [f"{z}/{y}/{x}" for z in range(grid[0]) for y in range(grid[1]) for x in range(grid[2])]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, keys))
    json.dump({"complete": True, "source": URL}, open(attrs_path, "w"))
    print(f"wrote {OUT} {meta['shape']}")


if __name__ == "__main__":
    main()
