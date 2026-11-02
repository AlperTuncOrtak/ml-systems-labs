from common import CONFIG, DATA, RESULTS
from datetime import datetime, timezone
from sklearn.datasets import fetch_openml


def ingest():
    """Read the raw Adult dataset exactly as published. No modification."""
    ds = CONFIG["dataset"]
    bunch = fetch_openml(name=ds["name"], version=ds["version"], as_frame=True,
                         data_home=DATA / "openml")
    df = bunch.frame
    n_rows, n_cols = df.shape
    ok = n_rows == ds["expected_rows"] and n_cols == ds["expected_cols"]
    log = (f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
           f"ingested adult v{ds['version']} | rows={n_rows} cols={n_cols} | "
           f"expected {ds['expected_rows']}x{ds['expected_cols']} -> {'OK' if ok else 'MISMATCH'}")
    (RESULTS / "ingestion_log.txt").write_text(log + "\n")
    print(log)
    if not ok:
        raise ValueError("Unexpected raw data shape")
    return df


if __name__ == "__main__":
    df = ingest()
    # quick look to design the validation step (read-only)
    print("\nColumns and dtypes:")
    print(df.dtypes.to_string())
    print("\nTarget values:")
    print(df[CONFIG["dataset"]["target"]].value_counts(dropna=False).to_string())
    print("\nMissing (NaN) per column:")
    print(df.isna().sum()[lambda s: s > 0].to_string())
    print("\nCells equal to '?':", int((df.astype(str) == "?").sum().sum()))
    print("Exact duplicate rows:", int(df.duplicated().sum()))
    print("\nNumeric ranges:")
    print(df.select_dtypes("number").agg(["min", "max"]).T.to_string())
