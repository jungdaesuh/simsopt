# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.0"]
# ///
import subprocess, shutil, sys
checks = ["cmake", "g++", "make", "git", "apt"]
for c in checks:
    path = shutil.which(c)
    print(f"{c}: {path or 'NOT FOUND'}")
try:
    r = subprocess.run(["apt", "list", "--installed"], capture_output=True, text=True)
    for pkg in ["boost", "eigen", "pybind11"]:
        matches = [l for l in r.stdout.split("\n") if pkg in l.lower()]
        for m in matches:
            print(f"  apt: {m.strip()}")
except FileNotFoundError:
    print("apt not available")
print(f"Python: {sys.version}")
# Check pip for build deps
r = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
for pkg in ["pybind11", "scikit-build", "cmake", "ninja"]:
    matches = [l for l in r.stdout.split("\n") if pkg in l.lower()]
    for m in matches:
        print(f"  pip: {m.strip()}")
