"""Real-device cache/dispatch regression. Run after test_solver.py on macOS."""
import os
import pathlib
import plistlib
import subprocess
import sys
import tempfile

exe = str(pathlib.Path(sys.argv[1]).resolve())
hill = pathlib.Path(__file__).with_name("fixtures")/"hill22.txt"
bounds = ["3314", "3762", "88", "92", "-678", "-230"]  # 1,008,005, non-divisible tail
with tempfile.TemporaryDirectory() as directory:
    cache = pathlib.Path(directory) / "nested" / "tune.plist"
    env = dict(os.environ, GRASS_METAL_AUTOTUNE_CACHE=str(cache))

    def run(mode="auto", extra=(), search=bounds):
        result = subprocess.run([exe, "--formation", str(hill), "--autotune", mode,
                                 *extra, *search], text=True, capture_output=True, env=env)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "3538 90 -454 0\n", result.stdout
        return result.stderr

    forced = run("force")
    assert "variants 16 verified_samples 144" in forced, forced
    assert "saved" in forced and cache.exists(), forced
    stored = plistlib.loads(cache.read_bytes())
    assert len(stored) == 1 and all(r["milliseconds"] > 0 for r in stored.values())
    auto = run()
    assert "cache hit" in auto and "autotune sample" not in auto, auto
    before = cache.read_bytes()
    assert "autotune off" in run("off") and cache.read_bytes() == before
    print("PASS force, disk cache hit, off, non-divisible batch tail; identical hill origin")
    print(forced.strip())
    print(auto.strip())

    # Shape differs while preserving the known origin. This must not reuse timings.
    altered = bounds.copy()
    altered[1] = "3763"
    assert "autotune selected" in run(search=altered)
    assert len(plistlib.loads(cache.read_bytes())) == 2
    print("PASS shape invalidation")

    assert "autotune selected" in run("force")  # force bypasses a valid entry
    cleared = run("off", ["--clear-autotune-cache"])
    assert "cache cleared" in cleared and not cache.exists()
    print("PASS force bypass and clear-cache")

    cache.write_bytes(b"corrupt plist")
    assert "autotune selected" in run()
    records = plistlib.loads(cache.read_bytes())
    for record in records.values():
        record["width"] = 0  # invalid cached launch must be rejected
    cache.write_bytes(plistlib.dumps(records))
    assert "autotune selected" in run()
    print("PASS corrupt plist and invalid launch rejection")

    other = pathlib.Path(directory) / "override.plist"
    assert "autotune selected" in run(extra=["--autotune-cache", str(other)])
    assert other.exists()
    # A non-directory parent must not stop the solver if saving fails.
    blocked = pathlib.Path(directory) / "not-a-directory"
    blocked.write_text("file")
    assert "not-saved" in run(extra=["--autotune-cache", str(blocked / "cache.plist")])
    print("PASS explicit cache path and unwritable-cache fallback")

    for extra in [["--autotune", "invalid"], ["--autotune"], ["--autotune-cache"]]:
        result = subprocess.run([exe, *extra], text=True, capture_output=True, env=env)
        assert result.returncode != 0
    print("PASS malformed autotune options")
