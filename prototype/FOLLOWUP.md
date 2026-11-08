Four additional unlabeled screenshots were evaluated without changing either detector. Screenshot order follows the user's upload order.

| Image / filename time suffix | Existing tiles / seconds | Grid-first tiles / seconds | Hybrid tiles / total seconds |
| --- | --- | --- | --- |
| 1 / 08.49 | 50 / 23.48 | 0 / 3.90 | 23 / 30.11 |
| 2 / 08.13 | 7 / 8.71 | 0 / 3.35 | 14 / 15.64 |
| 3 / 08.29 | 11 / 11.25 | 0 / 3.94 | 9 / 19.82 |
| 4 / 07.12 | 4 / 7.70 | 0 / 4.25 | 11 / 16.16 |

Hybrid totals include the required baseline pass. Counts do not establish rotation accuracy; none of these screenshots has supplied labels.

The best formation is selected by the most accepted tiles, without using candidate search results to choose a favorable subset. That selects existing detector for images 1 and 3, hybrid for images 2 and 4.

Search uses the CURRENT WEB UI default inclusive bounds x/z -2000 through 2000 and y -64 through 320, with all four north orientations, for 24,652,321,540 origins/orientations per image. This differs from the native CUDA/Metal CLI's wider defaults. Coordinates are the exported formation's local anchor, not a guaranteed camera/player position.

scan-default.cpp implements the repository's exact wrapped Java hash with AVX-512 and two CPU threads in the sandbox. It does not touch the desktop GPU or any suspended desktop jobs. Before searching, all 1,024 SIMD hash outputs matched an independent Python hash oracle across random int32 x/z and negative heights. Four small all-north exhaustive searches also matched the independent scalar oracle exactly, including injected known answers. verify-candidates.cjs cross-checks every returned candidate with the unchanged production solver.wasm. No candidate is treated as screenshot ground truth merely because it satisfies the hash.

All four full searches completed. Image 1 has exactly one candidate, (-525, 65, 143), north 2, satisfying all 50 observations. Image 2 has 93 candidates with its 14-tile hybrid. Image 3 has 5,816 candidates with 11 existing-detector tiles. Image 4 has 5,896 candidates with 11 hybrid tiles. Every candidate from every image passes the unchanged production WASM solver. Search times were 49.43, 49.02, 49.24 and 50.99 seconds respectively. Uniqueness applies only to these bounds and this formation, not the whole world or independent screenshot truth. Full coordinate/north lists are retained in followup-results/solver.json.

Build and test the search harness on an AVX-512-capable x86 CPU.

```sh
g++ -O3 -std=c++17 -mavx512f -mavx512dq -fopenmp \
  prototype/scan-default.cpp -o /tmp/grass-scan
python prototype/test_scan_default.py /tmp/grass-scan
/tmp/grass-scan formation.txt -2000 2000 -64 320 -2000 2000 2
```

The standalone line/grid proposal stage still finds zero reliable seeds in all four. Yellow biome tint is not proven to be its principal problem, since its classifier accepts the hybrid's seed quads. The hybrid helps two screenshots but loses evidence in the other two. Retain the existing detector; do not port this grid-first proposal stage as a replacement.

All 12 overlays, plus a contact sheet, are hosted as isolated sandbox artifacts, not on grass.kjt.lol. No push, merge or application deployment was performed.
