The grid-first prototype does not beat the existing detector on the failing plains screenshot. Do not port or ship it as a replacement.

Two complete runs produced identical tile counts. Times below are from the second run on the same CPU sandbox, OpenCV 4.13.0, one OpenCV thread, Node 24.22.2, production source 0580d5970dca983456f5a3f129e1a50386bfad95. They are measured wall times, not M1 or browser performance claims. Python times include proposal generation, classification and lattice growth; baseline times cover runDetection after module initialization.

| Image | Existing tiles / seconds | Grid-only tiles / seconds | Hybrid tiles / total seconds |
| --- | --- | --- | --- |
| Hill | 22 / 13.25 | 0 / 4.02 | 5 / 18.76 |
| Birch | 78 / 45.47 | 33 / 4.44 | 23 / 50.79 |
| Overhead | 15 / 19.17 | 46 / 3.86 | 36 / 23.24 |
| New plains | 2 / 5.42 | 0 / 3.40 | 2 / 10.06 |

Hybrid total includes the entire baseline detection plus the extra Python stage, not just the additional work.

Hill baseline rotations match all 22 labeled faces, and all 22 satisfy the independent world hash. Grid-only finds no hill formation. Hybrid matches 5/5 labeled rotations but loses most evidence. Both baseline and hybrid find the known 3538,90,-454 after origin registration in the actual WASM solver. Baseline has one result in the bounded neighborhood; the five-face hybrid also admits a second result, so retaining the known answer alone is not a sufficient success criterion.

Birch baseline matches 34/34 overlapping labeled rotations, grid-only 29/29, and hybrid 20/20. These fractions describe labeled overlap, not total precision. Independent hash checks pass all 78 baseline, 33 grid-only and 23 hybrid observations after geometry-only origin registration. Each mode finds only -4133,63,-2049 in the bounded solver neighborhood. No full-world uniqueness claim is made.

Overhead has no absolute-coordinate or independently labeled rotation ground truth. Its count increase is promising for a specialized fronto-parallel fallback, not proof of higher accuracy. Plains also lacks labeled rotations or known coordinates. Two high-confidence tops are not enough to claim useful or unique cracking. Neither experimental mode adds a verified plains tile over the baseline.

The plains line hypotheses are dominated by clutter and do not yield a grass-plane template proposal above the coarse threshold. Period estimation is fragile when only a small, occluded or strongly foreshortened area is rectified. Low angle alone is not proven to be the only failure cause.

The hybrid variant can even reduce the largest plane because duplicate seed hypotheses affect the greedy plane grouping and fitted homography. That is an experimental defect, not an improvement to hide behind the larger overhead count. Relative alignment across independent height levels is also unfinished; merging all planes into one y=0 formation would be incorrect.

Ten unit tests pass, including all four rotations under tint and blur, small and large occluders, clipping, blank images, noise rejection, under-resolved faces and keeping independent planes separate. Synthetic tests do not establish real-world accuracy.

Keep the existing browser detector. A subsequent experiment should first stabilize plane selection with nonduplicate, spatially distributed evidence and joint camera constraints, then optimize multi-scale texture alignment within those constraints. Evaluate any semantic occluder mask separately. A browser port is not warranted until the plains case gains enough independently verified observations without losing hill and birch coverage.
