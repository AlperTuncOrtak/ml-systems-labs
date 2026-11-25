import sys
import platform
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

REQ = Path(__file__).resolve().parents[2] / "requirements.txt"

print(f"Python {sys.version}")
print(f"Platform {platform.platform()}")
for line in REQ.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name = line.split("==")[0]
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")
