import argparse
import numpy as np
import os
import torch
from omni.isaac.lab.app import AppLauncher
from isaaclab.assets import AssetBaseCfg
# 1. Launch the Isaac Sim app first (Required boilerplate)
app_launcher = AppLauncher(argparse.ArgumentParser().parse_args())
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass
from isaaclab_assets.robots.crazyflie import CRAZYFLIE_CFG

# 2. Import the spikelab-jhu plugin components
from dvs_gen.sensors import DVSCameraCfg, DVSCamera 



@configclass
class DroneTrackingSceneCfg(InteractiveSceneCfg):
    """Configures the simulation scene with two drones and an event camera."""
    
# 0. Load the Room Environment correctly
    warehouse = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/warehouse",
        spawn=sim_utils.UsdFileCfg(
            # Hardcode the path or pass it in, but don't use dynamic omni functions here
            usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        )
    )

    # 1. Create a Default Light Source correctly
    distant_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(1.0, 1.0, 1.0)
        )
    )
    
# 1. Update the Observer Drone Config
    observer_cfg = CRAZYFLIE_CFG.copy()
    observer_cfg.prim_path = "{ENV_REGEX_NS}/Observer"
    # The observer is just background as far as the algorithm is concerned
    observer_cfg.spawn.semantic_tags = [("class", "observer")] 
    observer = observer_cfg

    # 2. Update the Target Drone Config
    target_cfg = CRAZYFLIE_CFG.copy()
    target_cfg.prim_path = "{ENV_REGEX_NS}/Target"
    # This is the critical tag your EVIMO algorithm needs to find
    target_cfg.spawn.semantic_tags = [("class", "enemy")] 
    target = target_cfg

    # The Event Camera mounted to the observer
    dvs_cam = DVSCameraCfg(
        prim_path="{ENV_REGEX_NS}/Observer/body/dvs_cam", # Mount to the Crazyflie's main body
        update_period=0.0, 
        height=480, 
        width=640, 
        threshold=0.15,
        enable_warp=True, # Critical: enables motion-vector extraction for the 5070
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 1e5)),
        data_types=["rgb", "depth", "motion_vectors", "semantic_segmentation"],
        offset=DVSCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0)) # Push slightly forward so it doesn't clip the drone frame
    )

def main():
    # Instantiate the scene
    scene_cfg = DroneTrackingSceneCfg(num_envs=1)
    scene = InteractiveScene(scene_cfg)
    scene.reset()

    # Create the simulation context (controls time and physics)
    sim_cfg = sim_utils.SimulationCfg(
    dt=1.0 / 20.0,  
    render_interval=1 
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 3. Wrap the camera to extract data to disk
    dvs = DVSCamera.from_scene(scene, ["dvs_cam"], out_dir="/tmp/drone_tracking_dvs")
    
    # Setup rendering and warp timing
    render_hz = 20 
    K = 8  # Warp multiplier: generates events at 400Hz (50 * 8)
    dt = 1.0 / render_hz
    dt_fine = 1.0 / (render_hz * K)
    
    prev_snap = dvs.snapshot()
    t_prev = 0.0

    # Create a folder for the 50Hz ground truth masks
    gt_dir = "/tmp/drone_tracking_dvs/ground_truth"
    os.makedirs(gt_dir, exist_ok=True)

    # --- ONE-TIME SEMANTIC ID LOOKUP ---
    # Fetch mapping once to avoid searching dict inside the main loop
    id_map = scene["dvs_cam"].data.info[0]["semantic_segmentation"]["classToLabels"]
    enemy_id = None
    for class_name, obj_id in id_map.items():
        if "enemy" in class_name:
            enemy_id = int(obj_id)
            break
            
    if enemy_id is None:
        print("[WARNING] 'enemy' semantic class not found in scene segmentations!")

    print("Starting simulation loop...")
    frame_count = 0

    while simulation_app.is_running():
        
    # --- 1. APPLY KINEMATIC TRAJECTORIES ---
        # We move the drones directly by setting their root states (Position + Quaternion)
        # Shape of root_pose: [num_envs, 7] -> (x, y, z, qw, qx, qy, qz)
        
        # Observer Drone: Fly slowly forward along the X-axis
        obs_pose = scene["observer"].data.default_root_state[:, :7].clone()
        obs_pose[:, 0] = 8 * torch.cos(t_prev * 0.5)
        obs_pose[:, 1] = 8 * torch.sin(t_prev * 0.5)

        obs_pose[:, 2] = 2.0  # Hover at 2 meters
        scene["observer"].write_root_pose_to_sim(obs_pose)
        
        # Target Drone: Fly in a circle 3 meters in front of the observer
        target_pose = scene["target"].data.default_root_state[:, :7].clone()
        target_pose[:, 0] = obs_pose[:, 0] + 3.0  # Always 3m in front
        target_pose[:, 1] = obs_pose[:, 1] + 1.5 * torch.sin(t_prev * 2.0) 
        target_pose[:, 2] = obs_pose[:, 2] + 1.0 * torch.cos(t_prev * 3.0)
        scene["target"].write_root_pose_to_sim(target_pose)

        # --- 2. STEP PHYSICS & RENDER (50Hz) ---
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=dt)
        
        # --- 3. EXTRACT GROUND TRUTH SEGMENTATION ---
        seg_tensor = scene["dvs_cam"].data.output["semantic_segmentation"][0]
        
        if enemy_id is not None:
            binary_mask = (seg_tensor == enemy_id).to(torch.uint8)
        else:
            binary_mask = torch.zeros_like(seg_tensor, dtype=torch.uint8)
                
        # 4. Create the binary mask
        if enemy_id is not None:
            # This creates an array of True/False, converted to 1/0
            binary_mask = (seg_tensor == enemy_id).cpu().numpy().astype(np.uint8)
        else:
            # Fallback if the enemy isn't visible in this frame
            binary_mask = np.zeros_like(seg_tensor.cpu().numpy(), dtype=np.uint8)

        # Save the mask frame
        timestamp_us = int(t_prev * 1e6)
        np.save(f"{gt_dir}/mask_{timestamp_us}.npy", binary_mask)

        # --- 4. SYNTHESIZE HIGH-FREQUENCY EVENTS (400Hz) ---
        cur_snap = dvs.snapshot()
        dvs.warp_and_process(prev_snap, cur_snap, K, t_prev, dt_fine)
        
        # Advance time
        prev_snap = cur_snap
        t_prev += dt
        frame_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()