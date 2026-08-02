import argparse
import numpy as np
import os
import torch
import math
import ast
import time
from isaaclab.app import AppLauncher

# Inject IsaacSim default arguments to parser
parser = argparse.ArgumentParser() 
AppLauncher.add_app_launcher_args(parser) 

args_cli = parser.parse_args() 
args_cli.enable_cameras = True 
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli) 
simulation_app = app_launcher.app 


from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass
# from isaaclab_assets.robots.crazyflie import CRAZYFLIE_CFG
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.actuators import ImplicitActuatorCfg
# 2. Import the spikelab-jhu plugin components
from dvs_gen.sensors import DVSCameraCfg, DVSCamera, tag_dvs_cameras

from isaacsim.storage.native import get_assets_root_path
assets_root_path = get_assets_root_path()


@configclass
class DroneTrackingSceneCfg(InteractiveSceneCfg):
    """Configures the simulation scene with two drones and an event camera."""

    env_spacing = 5.0  # Spacing distance between environments (meters)
# 0. Load the Room Environment correctly
    warehouse = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/warehouse",
        spawn=sim_utils.UsdFileCfg(
            # Hardcode the path or pass it in, but don't use dynamic omni functions here
            usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
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


    # 2. Update the Target Drone Config
    observer_cfg = CRAZYFLIE_CFG.copy()
    observer_cfg.prim_path = "{ENV_REGEX_NS}/Target"
    observer_cfg.actuators = {"dummy_actuator": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)}  
    
    # --- NEW FIX: MAKE DRONE KINEMATIC ---
    # This overwrites the default physics and tells PhysX to ignore gravity and collisions
    observer_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True
    )
    # -------------------------------------

    # This is the critical tag your EVIMO algorithm needs to find
    observer_cfg.spawn.semantic_tags = [("class", "observer")] 
    observer = observer_cfg


# 2. Update the Target Drone Config
    target_cfg = CRAZYFLIE_CFG.copy()
    target_cfg.prim_path = "{ENV_REGEX_NS}/Target"
    target_cfg.actuators = {"dummy_actuator": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)}  
    
    # --- NEW FIX: MAKE DRONE KINEMATIC ---
    # This overwrites the default physics and tells PhysX to ignore gravity and collisions
    target_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True
    )
    # -------------------------------------

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


    # Create the simulation context (controls time and physics)
    sim_cfg = sim_utils.SimulationCfg(
    dt=1.0 / 20.0,  
    render_interval=1 
    )
    sim = sim_utils.SimulationContext(sim_cfg)

        # Instantiate the scene
    scene_cfg = DroneTrackingSceneCfg(num_envs=1)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    tag_dvs_cameras(scene, ["dvs_cam"]) # tag the cam for visualization
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
# --- ONE-TIME SEMANTIC ID LOOKUP ---
    # Before the while loop starts
    enemy_id = None
    print("Starting simulation loop...")
    frame_count = 0


    while enemy_id is None and simulation_app.is_running():

        # 1. Step the simulation so the renderer actually draws the frame
        sim.step()
        simulation_app.update()
        scene.update(dt=dt)

        cam_data = scene["dvs_cam"].data
        if cam_data.info is not None and len(cam_data.info) > 0:
            info = cam_data.info[0]
            if "semantic_segmentation" in info:
                seg_info = info["semantic_segmentation"]
                if "idToLabels" in seg_info:
                    for obj_id_str, class_info in seg_info["idToLabels"].items():
                        if "enemy" in str(class_info):
                            # Check if Replicator is giving us a color tuple or an integer
                            if obj_id_str.startswith("("):
                                enemy_id = ast.literal_eval(obj_id_str) # Converts string to tuple
                                print("it's in tuple format")
                            else:
                                enemy_id = int(obj_id_str) # Standard integer
                                print("it's in int format")
                                
                            print(f"[INFO] Found 'enemy' tag at frame {frame_count} with Identifier: {enemy_id}")
                            break
   

    while simulation_app.is_running():
        try:
            # --- 1. APPLY KINEMATIC TRAJECTORIES ---
            obs_pose = scene["observer"].data.default_root_state[:, :7].clone()
            obs_pose[:, 0] = 8 * math.cos(t_prev * 0.5)
            obs_pose[:, 1] = 8 * math.sin(t_prev * 0.5)
            obs_pose[:, 2] = 2.0  # Hover at 2 meters
            scene["observer"].write_root_pose_to_sim(obs_pose)
            
            target_pose = scene["target"].data.default_root_state[:, :7].clone()
            target_pose[:, 0] = obs_pose[:, 0] + 2.0  # Always 2m in front
            target_pose[:, 1] = obs_pose[:, 1] + 0.5 * math.sin(t_prev)
            target_pose[:, 2] = obs_pose[:, 2] + 0.25 * math.cos(t_prev)
            scene["target"].write_root_pose_to_sim(target_pose)

            # --- 2. STEP PHYSICS & RENDER (50Hz) ---
            scene.write_data_to_sim()
            sim.step()
            simulation_app.update()
            scene.update(dt=dt)
            
            cam_data = scene["dvs_cam"].data
            

            # --- 3. DYNAMIC SEMANTIC ID LOOKUP ---
          
                
            
            # --- 4. EXTRACT GROUND TRUTH SEGMENTATION ---
            seg_tensor = cam_data.output["semantic_segmentation"][0]
            
            if isinstance(enemy_id, tuple):
                # It's an RGBA color mask. We need to match all 4 channels (dim=-1)
                color_tensor = torch.tensor(enemy_id, device=seg_tensor.device, dtype=seg_tensor.dtype)
                # Check where the image pixels exactly match the RGBA color
                binary_mask = (seg_tensor == color_tensor).all(dim=-1).cpu().numpy().astype(np.uint8)
            else:
                # It's a standard 1D integer mask
                binary_mask = (seg_tensor == enemy_id).cpu().numpy().astype(np.uint8)
        

            # Save the mask frame
            timestamp_us = int(t_prev * 1e6)
            np.save(f"{gt_dir}/mask_{timestamp_us}.npy", binary_mask)

            # --- 5. SYNTHESIZE HIGH-FREQUENCY EVENTS (400Hz) ---
            cur_snap = dvs.snapshot()
            dvs.warp_and_process(prev_snap, cur_snap, K, t_prev, dt_fine)
            
            # Advance time
            prev_snap = cur_snap
            t_prev += dt
            frame_count += 1

            time.sleep(0.01)  

        except Exception as e:
            # If ANY Python error happens, catch it and print it immediately!
            import traceback
            print("\n" + "="*50)
            print("🚨 CRASH DETECTED IN SIMULATION LOOP 🚨")
            print("="*50)
            traceback.print_exc()
            print("="*50 + "\n")
            break # Break the loop so it can shut down cleanly

if __name__ == "__main__":
    main()
    simulation_app.close()