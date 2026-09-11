import h5py
import numpy as np
import os
import argparse
import glob
import json

def convert_isaac_to_evimo_v1(input_dir, output_dir, seq_name='sequence_00'):
    """
    Converts spikelab-jhu HDF5 events and NumPy masks into the strictly 
    compressed EVIMO v1 / EVIMO2v1 NPZ format.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    in_h5_path = os.path.join(input_dir, 'env0_ep0.h5')
    if not os.path.exists(in_h5_path):
        print(f"Error: {in_h5_path} not found.")
        return

    # 1. Prepare Events Array
    print(f"Loading events from {in_h5_path}")
    with h5py.File(in_h5_path, 'r') as f_in:
        # Load as float64 to preserve timestamp precision when we merge into a single array
        x = np.array(f_in['DVS/dvs_cam/x'], dtype=np.float64)
        y = np.array(f_in['DVS/dvs_cam/y'], dtype=np.float64)
        p = np.array(f_in['DVS/dvs_cam/p'], dtype=np.float64)
        t = np.array(f_in['DVS/dvs_cam/t'], dtype=np.float64) 

    if t.max() > 10000:
        if t.max() > 1e11:
            print("Detected nanoseconds. Converting to seconds...")
            t = t / 1e9
        else:
            print("Detected microseconds. Converting to seconds...")
            t = t / 1e6
        
    # EVIMO v1 requires an (N, 4) shape array where columns are strictly: [timestamp, x, y, polarity]
    events_arr = np.column_stack((t, x, y, p))
    print(f"  -> Formatted {len(events_arr)} events. Shape: {events_arr.shape}")

 # 2. Prepare Masks and Metadata
    mask_files = glob.glob(os.path.join(input_dir, 'ground_truth', 'mask_*.npy'))
    if not mask_files:
        print("Warning: No ground truth masks found in input directory.")
        return

    # --- NEW FIX: Sort numerically by timestamp, not alphabetically ---
    def get_timestamp_from_filename(filepath):
        base = os.path.basename(filepath)
        ts_str = base.replace('mask_', '').replace('.npy', '')
        return float(ts_str)

    mask_files.sort(key=get_timestamp_from_filename)
    # ----------------------------------------------------------------

    print(f"Found {len(mask_files)} ground truth masks. Bundling into sequence...")
    
    masks = []
    frames_meta = []
    
    for i, mf in enumerate(mask_files):
        ts_val = get_timestamp_from_filename(mf)
        
        # --- NEW FIX: Correct microsecond conversion threshold ---
        # If the value is larger than 1000, it's definitely in microseconds
        if ts_val > 1000:  
            ts_val /= 1e6 
        
        mask_data = np.load(mf).astype(np.uint16)
        mask_data = (mask_data > 0).astype(np.uint8) 
        
        masks.append(mask_data)
        
        frames_meta.append({
            'id': i,
            'timestamp': ts_val,
        })
        
    masks_arr = np.stack(masks, axis=0)
    print(f"  -> Formatted masks. Shape: {masks_arr.shape}")
    
    # Construct the meta dictionary
    meta_dict = {
        'frames': frames_meta
    }

# 3. Reformat arrays exactly as ev-loader expects for .h5
    t = events_arr[:, 0]
    x = events_arr[:, 1]
    y = events_arr[:, 2]
    p = events_arr[:, 3]
    
    events_arr_h5 = np.column_stack((x, y, t, p))
    events_t = t
    
    mask_timestamps_full = np.array([f['timestamp'] for f in frames_meta], dtype=np.float64)
    
    # --- NEW FIX: Filter out early frames with no events ---
    valid_indices = []
    for i, ts in enumerate(mask_timestamps_full):
        # Ensure there are at least 100 events BEFORE this frame's timestamp
        events_before = np.searchsorted(t, ts)
        if events_before > 100:  
            valid_indices.append(i)
            
    print(f"Dropping {len(mask_timestamps_full) - len(valid_indices)} early frames with zero event history.")
    
    # Apply the filter to arrays
    masks_arr = masks_arr[valid_indices]
    mask_timestamps = mask_timestamps_full[valid_indices]
    
    # Update frames_meta and fix their internal IDs
    valid_frames_meta = [frames_meta[i] for i in valid_indices]
    for new_id, frame in enumerate(valid_frames_meta):
        frame['id'] = new_id
    # -------------------------------------------------------
    
    print("Calculating event-to-frame index mapping...")
    index_arr = np.searchsorted(t, mask_timestamps).astype(np.int64)
    
    # 4. Setup Camera Parameters
    height = 260
    width = 346
    fx = 320.0
    fy = 320.0
    cx = 320.0
    cy = 240.0
    
    meta_for_string = {
        'meta': {
            'res_x': height, 'res_y': width,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
        },
        'frames': valid_frames_meta
    }

    # --- ADD THIS DEBUG BLOCK ---
    print("\n--- TIMESTAMP DEBUG INFO ---")
    print(f"Events total: {len(t)}")
    print(f"Event Time Range: {t[0]:.6f} to {t[-1]:.6f}")
    print(f"Mask Time Range:  {mask_timestamps[0]:.6f} to {mask_timestamps[-1]:.6f}")
    print(f"First 5 Mask Times: {mask_timestamps[:5]}")
    print(f"Are Events perfectly sorted? : {np.all(np.diff(t) >= 0)}")
    print("----------------------------\n")
    # ----------------------------

# 5. Save to HDF5
    out_h5_path = os.path.join(output_dir, f'{seq_name}.h5')
    print(f"Saving EVIMO formatted dataset to {out_h5_path}")
    
    with h5py.File(out_h5_path, 'w') as f_out:
        # Event data
        f_out.create_dataset("events", data=events_arr_h5, compression='gzip')
        f_out.create_dataset("events_t", data=events_t)
        
        # Mask and timing data
        f_out.create_dataset("mask", data=masks_arr, compression='gzip')
        f_out.create_dataset("index", data=index_arr)
        f_out.create_dataset("ts", data=mask_timestamps)
        
        # --- DUMMY DATA FOR STRICT DATALOADER ---
        print("Generating dummy depth, classical, and discretization frames...")
        dummy_depth = np.zeros_like(masks_arr, dtype=np.float32)
        dummy_classical = np.zeros((*masks_arr.shape, 3), dtype=np.uint8)
        dummy_discretization = np.zeros((1,), dtype=np.float32) # Tiny dummy array
        
        f_out.create_dataset("depth", data=dummy_depth, compression='gzip')
        f_out.create_dataset("classical", data=dummy_classical, compression='gzip')
        f_out.create_dataset("discretization", data=dummy_discretization)
        # ----------------------------------------
        
        # Camera intrinsic parameters
        f_out.create_dataset("height", data=height)
        f_out.create_dataset("width", data=width)
        f_out.create_dataset("fx", data=fx)
        f_out.create_dataset("fy", data=fy)
        f_out.create_dataset("cx", data=cx)
        f_out.create_dataset("cy", data=cy)
        
        # Meta dictionary as a string representation
        f_out.create_dataset("meta", data=np.string_(str(meta_for_string)))
    
    print(f"  -> Saved archive to {out_h5_path}")
    print("Conversion complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Isaac Sim DVS to EVIMO v1 .npz Format')
    parser.add_argument('--input_dir', type=str, default='/tmp/drone_tracking_dvs', help='Dir with events.h5 and ground_truth/')
    parser.add_argument('--output_dir', type=str, default='/home/vboxuser/algo_data/test/sequence_00', help='Output directory')
    parser.add_argument('--seq_name', type=str, default='sequence_00', help='Name of the output sequence')
    args = parser.parse_args()
    
    convert_isaac_to_evimo_v1(args.input_dir, args.output_dir, args.seq_name)