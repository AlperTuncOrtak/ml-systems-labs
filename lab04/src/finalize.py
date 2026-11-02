from common import CONFIG, DATA, LAB, RESULTS
import hashlib
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
import yaml

FILES = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(raw_df):
    processed = DATA / "processed"
    files = {f: sha256_file(processed / f) for f in FILES}
    combined = hashlib.sha256("".join(files[f] for f in FILES).encode()).hexdigest()
    raw_hash = hashlib.sha256(
        pd.util.hash_pandas_object(raw_df, index=True).values.tobytes()).hexdigest()
    path = RESULTS / "data_manifest.yaml"
    previous = yaml.safe_load(path.read_text())["processed_sha256"] if path.exists() else None
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "generated_utc": now,
        "config": CONFIG,
        "schema_sha256": sha256_file(LAB / "schema.yaml"),
        "raw_sha256": raw_hash,
        "processed_sha256": combined,
        "processed_files": {f: {"sha256": files[f],
                                "shape": list(np.load(processed / f).shape)} for f in FILES},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    status = "first run" if previous is None else ("MATCH" if previous == combined else "DIFFERENT")
    line = f"{now} | processed_sha256={combined} | vs previous run: {status}"
    with open(RESULTS / "reproducibility_log.txt", "a") as f:
        f.write(line + "\n")
    print(line)
    return combined
