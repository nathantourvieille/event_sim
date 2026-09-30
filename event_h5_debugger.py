import os
import glob
import argparse
import json
import numpy as np
import h5py

def parse_ts_value(name):
    try:
        return float(name.split("_")[-1].split(".")[0])
    except Exception:
        return None

def infer_time_scale(t):
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return "empty"
    maxv = np.max(np.abs(t))
    if maxv > 1e9:
        return "nanoseconds"
    if maxv > 1e6:
        return "microseconds"
    return "seconds"

def summarize_event_file(h5_path):
    print(f"\n=== EVENT FILE: {h5_path} ===")
    with h5py.File(h5_path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        if "DVS" not in f:
            print("No DVS group found.")
            return
        dvs = f["DVS"]
        print("DVS keys:", list(dvs.keys()))
        if "dvs_cam" not in dvs:
            print("No dvs_cam group found.")
            return
        cam = dvs["dvs_cam"]
        print("dvs_cam keys:", list(cam.keys()))

        for k in ["x", "y", "p", "t"]:
            if k in cam:
                arr = np.asarray(cam[k])
                print(f"\n[{k}]")
                print("  shape:", arr.shape)
                print("  dtype:", arr.dtype)
                print("  min:", np.min(arr) if arr.size else None)
                print("  max:", np.max(arr) if arr.size else None)
                if k in ["x", "y"]:
                    print("  unique count:", np.unique(arr).size)
            else:
                print(f"Missing key: {k}")

        x = np.asarray(cam["x"])
        y = np.asarray(cam["y"])
        p = np.asarray(cam["p"])
        t = np.asarray(cam["t"])

        print("\nEvent sanity checks:")
        print("  len(x) == len(y) == len(p) == len(t):", x.size == y.size == p.size == t.size)
        print("  x range:", float(np.min(x)) if x.size else None, float(np.max(x)) if x.size else None)
        print("  y range:", float(np.min(y)) if y.size else None, float(np.max(y)) if y.size else None)
        print("  polarity unique:", np.unique(p))
        print("  t scale guess:", infer_time_scale(t))
        if t.size:
            print("  t min/max:", float(np.min(t)), float(np.max(t)))
            print("  sorted ascending:", bool(np.all(np.diff(t) >= 0)))
            print("  first 10 timestamps:", t[:10])

def summarize_masks(mask_dir):
    print(f"\n=== MASKS: {mask_dir} ===")
    files = sorted(glob.glob(os.path.join(mask_dir, "mask_*.npy")))
    print("mask files:", len(files))
    for f in files[:10]:
        print(" ", os.path.basename(f))
    if not files:
        print("No mask files found.")
        return

    shapes = []
    ts_values = []
    for f in files:
        arr = np.load(f)
        shapes.append(arr.shape)
        ts = parse_ts_value(os.path.basename(f))
        ts_values.append(ts)

    print("shape set:", sorted(set(tuple(s) for s in shapes)))
    print("first 10 timestamps from filenames:", ts_values[:10])
    print("all same shape:", all(s == shapes[0] for s in shapes))

    arr0 = np.load(files[0])
    print("dtype of first mask:", arr0.dtype)
    print("min/max of first mask:", np.min(arr0), np.max(arr0))
    print("unique values of first mask:", np.unique(arr0)[:20])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--event_file", type=str, default="env0_ep0.h5")
    args = parser.parse_args()

    h5_path = os.path.join(args.input_dir, args.event_file)
    mask_dir = os.path.join(args.input_dir, "ground_truth")

    if not os.path.exists(h5_path):
        print(f"Missing event file: {h5_path}")
    else:
        summarize_event_file(h5_path)

    if os.path.exists(mask_dir):
        summarize_masks(mask_dir)
    else:
        print(f"\nMask dir not found: {mask_dir}")

    print("\n=== Likely checks to make ===")
    print("1) If t max is huge, timestamps are probably ns/us; convert to seconds before matching to masks.")
    print("2) x and y should usually be within image size. If they are much larger, your camera shape/resolution is wrong.")
    print("3) If polarity values are not -1/+1 or 0/1, check the EVIMO loader expectations.")
    print("4) Mask filenames are only as good as the timebase used when saving them; they need to match event timestamps.")
    print("5) If event times are sorted but mask times are not, the timeline is inconsistent.")

if __name__ == "__main__":
    main()