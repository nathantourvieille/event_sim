# This is a 1 sec RMSE test to see how accurate this event cam sim is

import numpy as np
import torch
# https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.simulation_app/docs/index.html
from isaacsim.simulation_app import SimulationApp

# Set headless to False if you want to see the Isaac Sim GUI viewport alongside OpenCV,
# or True for maximum performance in a pure CLI terminal environment.
CONFIG = {
    "headless": False,
    "width": 1920,
    "height": 1080,
}
simulation_app = SimulationApp(CONFIG)

import omni.replicator.core as rep

from sim_orchestrator import setup_isaac_environment, configure_carb_settings, update_camera_position

max_pixel = 1
sim_time = 0
HEIGHT = 340
WIDTH = 640

# 1. Setup env with moving drone and cam. speed should be a param
camera_path = setup_isaac_environment()

# 2.Create the Render Product and attach the Annotator
render_product = rep.create.render_product(camera_path, resolution=(WIDTH, HEIGHT))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda")
motion_annotator = rep.AnnotatorRegistry.get_annotator("motion_vectors", device="cuda")
rgb_annotator.attach([render_product])
motion_annotator.attach([render_product])


adaptive_timestamps = []
adaptive_values = []

# Initialize the first adaptive trigger
next_adaptive_t = 0.0

# Takes rgb array in cuda and the x, y coords and return log intensity of them
def get_log_vals(y, x, rgb):
    # Convert to grey
    gray_vals = (rgb[:, 0] * 0.299 + 
                 rgb[:, 1] * 0.587 + 
                 rgb[:, 2] * 0.114)
    
    log_intensities = torch.log(gray_vals + 1e-5)

    return log_intensities

# Moves the camera to set time and renders the frame and returns it
def move(camera_path, time):

    update_camera_position(camera_path, current_t)
    simulation_app.update() 
    rep.orchestrator.step(delta_time=0.0, wait_for_render=True, pause_timeline=True, rt_subframes=-1)
    rgb_data = rgb_annotator.get_data(device="cuda", do_array_copy=False)
    return rgb_data

def evaluate_on_gpu(gt_t, gt_vals, adapt_t, adapt_vals):
    # 1. Find the interval each ground-truth timestamp falls into. idx should look like: 1, 1, 1... 2, 2... Equal values will be placed to the left
    idx = torch.bucketize(gt_t, adapt_t, right=False)
    
    # 2. Clamp indices to the valid range to prevent out-of-bounds at the edges. Just in case the first vals of idx are 0
    idx = torch.clamp(idx, 1, len(adapt_t) - 1)
    
    # 3. Retrieve the bounding timestamps of rendered frames
    t0 = adapt_t[idx - 1]
    t1 = adapt_t[idx]
    
    # 4. Calculate interpolation weights (add 1e-8 to prevent division by zero)
    weight = (gt_t - t0) / torch.clamp(t1 - t0, min=1e-8)
    
    # Unsqueeze weight from (10000,) to (10000, 1) to broadcast across the 100 pixels
    weight = weight.unsqueeze(1)
    
    # 5. Retrieve the bounding values
    v0 = adapt_vals[idx - 1]
    v1 = adapt_vals[idx]
    
    # 6. Linearly interpolate the high-frequency signal
    reconstructed_vals = v0 + weight * (v1 - v0)
    
    # 7. Compute RMSE
    mse = torch.mean((reconstructed_vals - gt_vals) ** 2)
    rmse = torch.sqrt(mse)
    
    return rmse, reconstructed_vals


# --- PASS 1: GROUND TRUTH ---

NUM_PIXELS = 100
DURATION = 1.0
SIGNAL_FREQ = 10000

gt_timestamps = np.arange(0.0, DURATION, DURATION / SIGNAL_FREQ)

# Pre-allocate your Ground Truth history directly in VRAM
gt_values = torch.zeros((len(gt_timestamps), NUM_PIXELS), device="cuda")

# Pick your 100 random pixel coordinates once, stored on GPU
y_coords = torch.randint(0, HEIGHT, (NUM_PIXELS,), device="cuda")
x_coords = torch.randint(0, WIDTH, (NUM_PIXELS,), device="cuda")

for idx, current_t in enumerate(gt_timestamps):

    rgb_data = move(camera_path, current_t)

    # Save directly into your GPU history tensor
    gt_values[idx] = get_log_vals(y_coords, x_coords, rgb_data)

# --- PASS 2: Adaptive Algo ---
sim_time = 0
current_t = 0
step_size = 0.01
adaptive_timestamps = torch.zeros((len(gt_timestamps) // 10,), device="cuda")
adaptive_values = torch.zeros((len(gt_timestamps) // 10, NUM_PIXELS), device="cuda")
idx = 0

while sim_time < DURATION:

    rgb_data = move(camera_path, current_t)

    # Append the timestamp and value to respective lists
    adaptive_timestamps[idx] = sim_time
    adaptive_values[idx] = get_log_vals(y_coords, x_coords, rgb_data)
    idx += 1

    # Get motion and compute max velocity and next timestamp (H, W, 2)
    motion_data = motion_annotator.get_data(device="cuda", do_array_copy=False)
    squared = motion_data ** 2
    magnitudes = torch.sqrt(squared[:, :, 0] + squared[:, :, 1])
    velocities_px_s = magnitudes / step_size
    max_vel = torch.max(velocities_px_s)
    step_size = max_pixel / max_vel

    sim_time += step_size

final_adaptive_timestamps = adaptive_timestamps[:idx]
final_adaptive_values = adaptive_values[:idx]

# --- EVALUATION ---
gt_timestamps_gpu = torch.from_numpy(gt_timestamps).cuda()
rmse, reconstructed_signal = evaluate_on_gpu(
    gt_timestamps_gpu, 
    gt_values, 
    final_adaptive_timestamps, 
    final_adaptive_values
)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

gt_signal_cpu = gt_values.cpu().numpy()
reconstucted_signal_cpu = reconstructed_signal.cpu().numpy()

# Plot the original signal
plt.plot(gt_timestamps, gt_signal_cpu, label='Original', color='blue', linewidth=2)

# Plot the reconstructed signal with a dashed line for clear comparison
plt.plot(gt_timestamps, reconstucted_signal_cpu, label='Reconstructed', color='orange', linestyle='--', linewidth=2)

plt.title('Original vs. Reconstructed Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()