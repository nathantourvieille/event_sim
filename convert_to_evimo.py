import h5py
import numpy as np
import os
import argparse
import glob

def convert_isaac_to_evimo_v1(input_dir, output_dir, seq_name='sequence_00'):
    """
    Converts spikelab-jhu HDF5 events and NumPy masks into the strictly 
    compressed EVIMO v1 / EVIMO2v1 NPZ format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    in_h5_path = os.path.join(input_dir, 'events.h5')
    if not os.path.exists(in_h5_path):
        print(f"Error: {in_h5_path} not found.")
        return

    # 1. Prepare Events Array
    print(f"Loading events from {in_h5_path}")
    with h5py.File(in_h5_path, 'r') as f_in:
        # Load as float64 to preserve timestamp precision when we merge into a single array
        x = np.array(f_in['events/x'], dtype=np.float64)
        y = np.array(f_in['events/y'], dtype=np.float64)
        p = np.array(f_in['events/p'], dtype=np.float64)
        t = np.array(f_in['events/t'], dtype=np.float64) 
        
    # EVIMO v1 requires an (N, 4) shape array where columns are strictly: [timestamp, x, y, polarity]
    events_arr = np.column_stack((t, x, y, p))
    print(f"  -> Formatted {len(events_arr)} events. Shape: {events_arr.shape}")

    # 2. Prepare Masks and Metadata
    mask_files = sorted(glob.glob(os.path.join(input_dir, 'ground_truth', 'mask_*.npy')))
    if not mask_files:
        print("Warning: No ground truth masks found in input directory.")
        return

    print(f"Found {len(mask_files)} ground truth masks. Bundling into sequence...")
    
    masks = []
    frames_meta = []
    
    for i, mf in enumerate(mask_files):
        # Extract timestamp from filename (e.g., 'mask_1500000.npy' -> 1500000)
        base = os.path.basename(mf)
        ts_str = base.replace('mask_', '').replace('.npy', '')
        
        # In EVIMO, timestamps are usually float seconds.
        # Fallback logic: if your Isaac Sim outputs large integer microseconds, convert to seconds.
        ts_val = float(ts_str) 
        if ts_val > 1e8:  
            ts_val /= 1e6 
        
        # Load mask
        mask_data = np.load(mf).astype(np.uint16)
        
        # EVIMO strictly specifies that object IDs in masks must be multiplied by 1000
        # If your Isaac Sim produces standard IDs (1, 2, 3), we scale them.
        # (Assuming 0 remains 0 for the background)
        mask_data = mask_data * 1000 
        masks.append(mask_data)
        
        # EVIMO v1 meta requires a 'frames' list mapping indices to timestamps
        frames_meta.append({
            'id': i,
            'timestamp': ts_val,
            # EVIMO also includes ground truth 'pos' (poses) here if you have them
        })
        
    masks_arr = np.stack(masks, axis=0)
    print(f"  -> Formatted masks. Shape: {masks_arr.shape}")
    
    # Construct the meta dictionary
    meta_dict = {
        'frames': frames_meta
    }

    # 3. Save to a single compressed .npz file (The EVIMO v1 standard)
    out_npz_path = os.path.join(output_dir, f'{seq_name}.npz')
    print(f"Saving EVIMO v1 formatted dataset to {out_npz_path}")
    
    np.savez_compressed(
        out_npz_path, 
        events=events_arr,                      # Extracts as events.npy
        mask=masks_arr,                         # Extracts as mask.npy
        meta=np.array(meta_dict, dtype=object)  # Extracts as meta.npy
    )
    
    print(f"  -> Saved archive to {out_npz_path}")
    print("Conversion complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Isaac Sim DVS to EVIMO v1 .npz Format')
    parser.add_argument('--input_dir', type=str, default='/tmp/drone_tracking_dvs', help='Dir with events.h5 and ground_truth/')
    parser.add_argument('--output_dir', type=str, default='/tmp/evimo_drone_data', help='Output directory')
    parser.add_argument('--seq_name', type=str, default='sequence_00', help='Name of the output sequence')
    args = parser.parse_args()
    
    convert_isaac_to_evimo_v1(args.input_dir, args.output_dir, args.seq_name)