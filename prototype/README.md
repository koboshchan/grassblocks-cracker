This is a local Python/OpenCV experiment, not a replacement browser detector. Nothing in the web app or deployment is changed.

The detector fits image-line vanishing points, proposes ground-plane homographies, rectifies local windows, estimates texel frequencies, and searches anisotropically scaled copies of all four grass-top rotations. Candidate tiles are sampled at 16×16 texel centers. Acceptance requires normalized correlation of at least 0.88, a rotation margin of at least 0.35, and agreement in all four quadrants. Per-channel affine fitting tolerates biome tint and exposure. Limited RGB residual masking rejects gross outliers; it is NOT semantic tall-grass or mob segmentation. Large occluders and under-resolved tiles are rejected.

Independent coplanar groups are retained in result.json. Their relative integer heights and translations are not proven, so formation.txt exports only the largest single-plane group rather than inventing cross-height registration. This is an important incomplete part of the proposed full detector.

Install the pinned headless requirements in a virtual environment, then run from the repository root.

```sh
python -m pip install -r prototype/requirements.txt
PYTHONPATH=prototype python -m robust_detect.cli screenshot.png \
  --texture web/public/grass_block_top.png --out prototype/runs/example
PYTHONPATH=prototype python -m unittest discover -s prototype -v
```

Output includes result.json, overlay.png and the unchanged formation.txt format, a count followed by x y z rotation is_side. All exported observations are grass tops with is_side=0. Coordinates are local, not absolute world coordinates. North is not inferred from a screenshot.

benchmark.py runs the unmodified production detector and both experimental modes. The baseline takes only screenshot pixels and the reference texture, with the existing OpenCV.js, SIFT and detector WASM artifacts. It never reads a saved camera, face list or labeled fixture. The optional hybrid mode rechecks live baseline seed quads using the Python classifier; its additional runtime must be added to the baseline run.

```sh
python prototype/benchmark.py \
  --birch /path/to/birch.png --overhead /path/to/overhead.png \
  --plains /path/to/plains.png \
  --assets /path/to/built/web \
  --birch-fixture /path/to/birch/detections.json \
  --esbuild /path/to/esbuild --node /path/to/node \
  --out /path/to/results
```

The assets directory must contain dist/vendor/opencv.js, grass-sift.js and grass-sift.wasm, plus artifacts/detect-wasm.wasm and solver.wasm. Use trusted local artifacts only. Image decode and dimensions are checked, but this prototype is not an upload service or a security sandbox for hostile images.

Labeled fixture geometry is loaded exclusively in evaluate.py after detection. It registers exported lattice origins to the known fixture origins without choosing translations by hash agreement. solver-regression.cjs then uses the actual production WASM solver over a bounded 17×5×17 neighborhood, not a full-world search. Hill and birch coordinates are compared after this explicit origin conversion. The raw automatic anchors differ from the saved fixtures by block offsets, which is not a rotation error.

RESULTS.md records the measured outcome and why this has not been ported.
