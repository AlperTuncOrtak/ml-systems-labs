"""Run the whole Lab 4 pipeline with one command:
    python3 lab04/src/run_pipeline.py
"""
import data_card
import finalize
import label_audit
import preprocess
from ingest import ingest


def main():
    print("== Steps 2-5: ingest, validate, split, clean, features, baseline ==")
    preprocess.run()
    print("\n== Step 6: label-quality audit ==")
    label_audit.run()
    print("\n== Step 7: hash, manifest, data card ==")
    raw = ingest()
    processed_hash = finalize.write_manifest(raw)
    data_card.write(raw, processed_hash)
    print("\nPipeline complete. Processed SHA256:", processed_hash)


if __name__ == "__main__":
    main()
