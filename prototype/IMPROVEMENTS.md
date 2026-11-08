Existing-detector improvement experiment

Local only. The deployed detector and its defaults are unchanged. improve.ts is a separate, opt-in research entry point; nothing imports it in the application.

The first grid-first prototype did not beat the existing detector. This experiment instead retains its SIFT seeds, dense normalized correlation, four-rotation margin checks, ECC refinement and voxel camera model.

Measured interventions

Earlier ECC refinement relaxes only the preliminary refinement gate. Dense final acceptance still applies. Plane refitting estimates a homography from already accepted quads, retaining integer origins. Single-height voxel refitting fixes a real limitation: the old stage stopped whenever all detections occupied one height, even though those accepted quads could improve calibration. Flat refits use a plane homography instead of an underdetermined 3D DLT. The observed height is restored in the camera translation, preserving the original y anchor.

Looser descriptor ratios can discover additional seeds, but also change which plane is dominant. They are not a standalone replacement: ratio .76 reduced hill from 22 to 4 and birch from 78 to 47. The multi-trial prototype retains the original detections and accepts additional trials only after at least three overlapping pixel quads agree on an integer x/y/z translation, with 90% translation support. Any overlapping rotation conflict rejects that trial. Coordinate search is NEVER involved in this selection.

A marginally lower final correlation threshold (.84 instead of .88; voxel .72 instead of .78) enables camera bootstrapping in the mixed screenshot. Rotational margin and quadrant voting are retained. One bad observation can invalidate a formation, so counts alone are not evidence of correctness. Solver validation is reported independently.

No extra color-normalization pass was retained. The detector already operates on the green channel with mean-subtracted, variance-normalized correlation, which is invariant to affine tint and lighting. A unit test confirms this. Biome tint alone did not explain the failing proposal geometry. No semantic grass/flower/mob segmentation, side-face recognition or new object model is claimed.

Reproduction

Bundle prototype/baseline.ts with esbuild, platform=node, format=esm. Use prototype/improve-benchmark.py with --images pointing to a JSON mapping image names to screenshot paths, --bundle to the bundle, --vendor to the saved OpenCV/SIFT assets, --artifact to the unchanged production detect-wasm.wasm, --texture to web/public/grass_block_top.png, and --out to a new directory. The runner sets IMPROVED=1. Unset it to run the original detector; DETECT_OPTIONS can independently select an ablation.

prototype/check-improvement.py performs evaluation in a separate process. Labeled fixture quads and known coordinates never enter detection. prototype/improvement-tests.ts tests unchanged defaults, affine tint invariance, geometric registration, preserved anchors, and weak/conflicting overlap rejection. The original Python prototype tests remain separate.

Limits

This is a deliberately expensive multi-pass prototype, not a browser-ready replacement. Trial-count inflation and the risk of an extra false observation must be weighed against candidate reduction. Runtime must include every trial, not just the fastest winning trial. No uniqueness claim extends outside inclusive web bounds x/z [-2000,2000], y [-64,320], all four norths. Returned coordinates are formation anchors, not player positions. Hill and birch have independently labeled geometry; the new screenshots do not.

Benchmark interpretation

The requested earlier 93 savanna and 5,896 lava candidates came from the earlier hybrid prototype, not the original SIFT detector. Strict old-versus-new comparisons use original SIFT counts (7 savanna, 4 lava) and separately report the hybrid figures. Counting extremely ambiguous formations uses scan-default.cpp --count-only, so the plains screenshot cannot fill disk with more than a billion candidate lines. This mode is checked against explicit candidate enumeration and the independent Python oracle.

The original low-angle plains shot remains unresolved at two tops. Rejected trials can detect more visual patches there, but cannot establish the three geometric overlaps needed to safely join them to the original lattice. They are not exported merely to inflate counts.

OpenCV RANSAC state advances between trials. Fresh-process ablations and the integrated, cached-seed multi-trial runner can therefore differ slightly. Integrated reruns, when available, take precedence over assembled ablation unions; the two are not falsely described as identical end-to-end timings.
