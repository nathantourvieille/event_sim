import os
import glob
import argparse
import json
import ast
import h5py
import numpy as np

def safe_decode_meta(value):
    if value is None:
        return None

    # HDF5 scalar/bytes/object strings
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except Exception:
            text = str(value)
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return text

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return safe_decode_meta(value.item())
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return value

    return value

def print_dataset_summary(name, obj):
    print(f"\n[{name}]")
    print(f"  shape: {obj.shape}")
    print(f"  dtype: {obj.dtype}")

    try:
        data = obj[()]
    except Exception as e:
        print(f"  <could not read> {e}")
        return

    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="replace")
        print(f"  sample: {text[:500]}")
        return

    arr = np.asarray(data)
    if arr.ndim == 0:
        print(f"  value: {arr.item()}")
        return

    if arr.size == 0:
        print("  empty")
        return

    if arr.dtype.kind in {"U", "S", "O"}:
        try:
            sample = arr.reshape(-1)[:5]
            print("  sample:", sample)
        except Exception:
            print("  sample:", data)
        return

    # Print first few entries for numeric data
    print("  sample:", arr.reshape(-1)[:10])

def print_meta_if_present(f):
    if "meta" in f:
        meta_obj = f["meta"]
        try:
            meta_data = meta_obj[()]
        except Exception as e:
            print(f"\n[meta] <could not read> {e}")
            return

        decoded = safe_decode_meta(meta_data)
        print("\n=== META DATA ===")
        if isinstance(decoded, (dict, list)):
            print(json.dumps(decoded, indent=2, default=str)[:8000])
        else:
            print(decoded)
        print("=== END META DATA ===")

def inspect_file(path):
    print(f"\n=== FILE: {path} ===")
    with h5py.File(path, "r") as f:
        print("keys:", list(f.keys()))
        print_meta_if_present(f)

        # Walk datasets
        def recurse(name, obj):
            if isinstance(obj, h5py.Dataset):
                print_dataset_summary(name, obj)
        f.visititems(recurse)

        # Event summary
        if "events" in f:
            ev = np.asarray(f["events"], dtype=np.float64)
            print("\n=== EVENT SUMMARY ===")
            print("shape:", ev.shape)
            print("dtype:", ev.dtype)
            print("first 10 rows:")
            print(ev[:10])
            if ev.shape[1] >= 4:
                x, y, t, p = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
                print("x range:", float(np.min(x)), float(np.max(x)))
                print("y range:", float(np.min(y)), float(np.max(y)))
                print("t range:", float(np.min(t)), float(np.max(t)))
                print("polarity unique:", np.unique(p))
                print("timestamps sorted:", bool(np.all(np.diff(t) >= 0)))
                print("event order guess: [x, y, t, p]")

        # Mask summary
        if "mask" in f:
            mask = np.asarray(f["mask"])
            print("\n=== MASK SUMMARY ===")
            print("shape:", mask.shape)
            print("dtype:", mask.dtype)
            print("unique values:", np.unique(mask)[:20])
            print("min/max:", float(np.min(mask)), float(np.max(mask)))

        # ts summary
        if "ts" in f:
            ts = np.asarray(f["ts"], dtype=np.float64)
            print("\n=== FRAME TIMESTAMPS ===")
            print("shape:", ts.shape)
            print("min/max:", float(np.min(ts)), float(np.max(ts)))
            print("first 10:", ts[:10])

        # index summary
        if "index" in f:
            idx = np.asarray(f["index"], dtype=np.int64)
            print("\n=== INDEX SUMMARY ===")
            print("shape:", idx.shape)
            print("min/max:", int(np.min(idx)), int(np.max(idx)))
            print("first 20:", idx[:20])

        # height / width
        for key in ["height", "width", "fx", "fy", "cx", "cy"]:
            if key in f:
                v = np.asarray(f[key])
                print(f"{key}: {v}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    inspect_file(args.file)

if __name__ == "__main__":
    main()