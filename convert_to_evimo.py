#!/usr/bin/env python3
"""
Convert Isaac Sim DVS output into the minimal EVIMO-style .h5 that EVIMOSequence reads.

Inputs
  input_dir/env0_ep0.h5             -> DVS/dvs_cam/{x,y,p,t}   (t already in seconds)
  input_dir/ground_truth/mask_<n>.npy
        n = sim step number (1, 2, 3, ...). Only the ORDER matters for sorting;
        the frame time in seconds is computed as n * sim_dt.

Output keys (14, all datasets, flat file)
  REAL (the loader reads their contents):
    events    (N,4) float32  [x, y, t, p]  p in {0,1}, t in seconds, sorted
    events_t  (N,)  float32
    ts        (F,)  float64  frame times in seconds
    mask      (F,H,W) float64  values {0, 2000}
    height, width   int64 scalars
  PLACEHOLDERS (the loader opens them by name but never reads them):
    index, depth    empty arrays
    meta            the text "{}"
    discretization, fx, fy, cx, cy   scalar 0.0
"""
import argparse
import glob
import os

import h5py
import numpy as np

MASK_VALUE = 2000.0


def convert(args):
    os.makedirs(args.output_dir, exist_ok=True)
    H, W = args.height, args.width

    # ---------------------------------------------------------------- events
    in_h5 = os.path.join(args.input_dir, args.event_file)
    if not os.path.exists(in_h5):
        raise FileNotFoundError(in_h5)
    print(f"Loading events from {in_h5}")
    with h5py.File(in_h5, "r") as f:
        g = f[args.dvs_group]
        x = np.asarray(g["x"], dtype=np.float32)
        y = np.asarray(g["y"], dtype=np.float32)
        p = np.asarray(g["p"])
        t = np.asarray(g["t"], dtype=np.float64)  # seconds
    if not np.all(np.diff(t) >= 0):
        print("Events not sorted -> sorting")
        order = np.argsort(t, kind="stable")
        t, x, y, p = t[order], x[order], y[order], p[order]
    p = (p > 0).astype(np.float32)  # EVIMO polarity {0,1}
    print(f"  {t.size} events, t: {t[0]:.4f} -> {t[-1]:.4f} s")

    # ----------------------------------------------------------------- masks
    mask_dir = args.mask_dir or os.path.join(args.input_dir, "ground_truth")
    mask_files = glob.glob(os.path.join(mask_dir, "mask_*.npy"))
    if not mask_files:
        raise FileNotFoundError(f"No mask_*.npy in {mask_dir}")
    step_no = lambda fp: int(os.path.basename(fp)[5:-4])  # "mask_12.npy" -> 12
    mask_files.sort(key=step_no)  # numeric order, so mask_10 comes after mask_9
    ts = np.array([step_no(fp) for fp in mask_files], dtype=np.float64) * args.sim_dt
    F, N = len(mask_files), t.size
    print(f"  {F} masks, ts: {ts[0]:.4f} -> {ts[-1]:.4f} s (step number x {args.sim_dt:g} s)")
    if ts[-1] > t[-1] + 1e-9:
        print("WARNING: last mask is later than the last event - check --sim_dt.")

    # ----------------------------------------------------------------- write
    out_h5 = os.path.join(args.output_dir, f"{args.seq_name}.h5")
    print(f"Writing {out_h5}")
    n_nonempty = 0
    with h5py.File(out_h5, "w") as fo:
        # REAL: events + events_t, written in slices so RAM never doubles
        ev = fo.create_dataset("events", shape=(N, 4), dtype=np.float32,
                               chunks=(min(N, 1 << 20), 4), compression="gzip", compression_opts=4)
        et = fo.create_dataset("events_t", shape=(N,), dtype=np.float32,
                               chunks=(min(N, 1 << 20),), compression="gzip", compression_opts=4)
        step = 5_000_000
        for s in range(0, N, step):
            e = min(s + step, N)
            ev[s:e] = np.column_stack((x[s:e], y[s:e], t[s:e], p[s:e])).astype(np.float32)
            et[s:e] = t[s:e].astype(np.float32)

        # REAL: frame times, masks (one frame at a time), image size
        fo.create_dataset("ts", data=ts)
        mk = fo.create_dataset("mask", shape=(F, H, W), dtype=np.float64,
                               chunks=(1, H, W), compression="gzip", compression_opts=4)
        for i, fp in enumerate(mask_files):
            m = np.load(fp)
            if m.shape != (H, W):
                raise ValueError(f"{fp} has shape {m.shape}, expected {(H, W)}")
            m = (m > 0).astype(np.float64) * MASK_VALUE
            n_nonempty += int(m.any())
            mk[i] = m
        fo.create_dataset("height", data=np.int64(H))
        fo.create_dataset("width", data=np.int64(W))

        # PLACEHOLDERS: must exist (the loader opens them by name) but are never read
        fo.create_dataset("index", shape=(0,), dtype=np.uint32)
        fo.create_dataset("depth", shape=(0,), dtype=np.float64)
        fo.create_dataset("meta", data=np.bytes_("{}"))
        for k in ("discretization", "fx", "fy", "cx", "cy"):
            fo.create_dataset(k, data=np.float64(0.0))

    print(f"Done. {N} events, {F} frames, non-empty masks: {n_nonempty}/{F}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Isaac Sim DVS output to a minimal EVIMO-style .h5")
    ap.add_argument("--input_dir", default="/media/vboxuser/T9/sim_data")
    ap.add_argument("--output_dir", default="/media/vboxuser/T9/sim_data/test/sequence_00")
    ap.add_argument("--seq_name", default="sequence_00")
    ap.add_argument("--event_file", default="env0_ep0.h5")
    ap.add_argument("--dvs_group", default="DVS/dvs_cam")
    ap.add_argument("--mask_dir", default=None, help="folder with mask_<n>.npy (default: input_dir/ground_truth)")
    ap.add_argument("--sim_dt", type=float, default=1.0 / 40, help="seconds per sim step (1/render_hz)")
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--width", type=int, default=346)
    convert(ap.parse_args())