import argparse
import numpy as np
import os
import torch
import math
import ast
import time
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from isaaclab.app import AppLauncher

# Inject IsaacSim default arguments to parser
parser = argparse.ArgumentParser() 
AppLauncher.add_app_launcher_args(parser) 

args_cli = parser.parse_args() 
args_cli.enable_cameras = True 
# Inject the extension commands into kit_args as a STRING
plugin_ext_folder = "/home/vboxuser/event_sim/isaac-sim-event-camera-plugin/extension"
custom_kit_args = f"--ext-folder {plugin_ext_folder} --enable dvs_preview"

# If nothing was passed, set it directly
if args_cli.kit_args is None:
    args_cli.kit_args = custom_kit_args
else:
    # If it's already a string, just add our commands to the end of it with a space
    args_cli.kit_args += f" {custom_kit_args}"

# 4. Launch the app
app_launcher = AppLauncher(args_cli) 
simulation_app = app_launcher.app


from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass
# from isaaclab_assets.robots.crazyflie import CRAZYFLIE_CFG
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.actuators import ImplicitActuatorCfg
from dvs_gen.sensors import DVSCameraCfg, DVSCamera, tag_dvs_cameras

from isaacsim.storage.native import get_assets_root_path
assets_root_path = get_assets_root_path()


@configclass
class DroneTrackingSceneCfg(InteractiveSceneCfg):
    """Configures the simulation scene with two drones and an event camera."""
    env_spacing = 5.0
    warehouse = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/warehouse",
        spawn=sim_utils.UsdFileCfg(
            # Hardcode the path or pass it in, but don't use dynamic omni functions here
            usd_path=f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        )
    )

    distant_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/DistantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(1.0, 1.0, 1.0)
        )
    )

    # 1. OBSERVER DRONE
    observer_cfg = CRAZYFLIE_CFG.copy()
    observer_cfg.prim_path = "{ENV_REGEX_NS}/Observer"
    observer_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
    observer_cfg.spawn.semantic_tags = [("class", "observer")]
    observer_cfg.init_state.pos = (0.0, 0.0, 2.0)
    observer = observer_cfg

    # 2. TARGET DRONE extra big for model
    target_cfg = CRAZYFLIE_CFG.copy()
    target_cfg.prim_path = "{ENV_REGEX_NS}/Target"
    target_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
    target_cfg.spawn.semantic_tags = [("class", "enemy")]
    target_cfg.spawn.scale = (3.0, 3.0, 3.0)  # Doubles the size in X, Y, and Z axes
    target_cfg.init_state.pos = (3.0, 0.0, 2.0)
    target = target_cfg

    # The Event Camera mounted to the observer
    dvs_cam = DVSCameraCfg(
        prim_path="{ENV_REGEX_NS}/Observer/body/dvs_cam", # Mount to the Crazyflie's main body
        update_period=0.0, 
        height=260, 
        width=346, 
        threshold=0.15,
        enable_warp=True, # Critical: enables motion-vector extraction for the 5070
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 1e5)),
        data_types=["rgb", "depth", "motion_vectors", "semantic_segmentation"],
        offset=DVSCameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.05), convention="world") # Push slightly forward so it doesn't clip the drone frame
    )



def main():
    # Setup rendering and warp timing
    render_hz = 40 # has to be a multiple of 40hz to be able to save GT at 40 hz
    K = 10  # Warp multiplier: generates events at 400Hz (50 * 8)
    dt = 1.0 / render_hz
    dt_fine = 1.0 / (render_hz * K)

    # Create the simulation context (controls time and physics)
    sim_cfg = sim_utils.SimulationCfg(dt=dt,  render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Instantiate the scene
    scene_cfg = DroneTrackingSceneCfg(num_envs=1)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    tag_dvs_cameras(scene, ["dvs_cam"]) # tag the cam for visualization
    # 3. Wrap the camera to extract data to disk
    dvs = DVSCamera.from_scene(scene, ["dvs_cam"], out_dir="/media/vboxuser/T9/sim_data/events")

    def _splat_batch(img, dx, dy, imp, hw=1.0, beta=12.0):
        """Batched softmax-splat: ``img`` (M,C,H,W); ``dx``,``dy``,``imp`` (M,H,W).
        All M independent splats run in ONE index_add (the batch index is folded into
        the scatter target). Returns out (M,C,H,W), holes (M,H,W)."""
        import math
        M, C, H, W = img.shape
        dev = img.device
        ys, xs = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        xs = xs.float(); ys = ys.float()
        tx = xs + dx                                       # (M,H,W)
        ty = ys + dy
        x0 = torch.floor(tx).long(); y0 = torch.floor(ty).long()
        imp_max = imp.flatten(1).amax(1).clamp(min=1e-6).view(M, 1, 1)
        imp_mult = torch.exp(beta * (imp / imp_max))       # (M,H,W) — foreground/fast wins
        HW = H * W
        color = torch.zeros(C, M * HW, device=dev)
        weight = torch.zeros(M * HW, device=dev)
        src = img.permute(1, 0, 2, 3).reshape(C, M * HW)
        boff = (torch.arange(M, device=dev) * HW).view(M, 1, 1)     # per-batch index offset
        r = int(math.ceil(hw))
        for ox in range(1 - r, 1 + r):
            for oy in range(1 - r, 1 + r):
                xx = x0 + ox; yy = y0 + oy
                wgt = (1 - (tx - xx.float()).abs() / hw).clamp(min=0) * \
                    (1 - (ty - yy.float()).abs() / hw).clamp(min=0)
                valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                wflat = (wgt * valid.float() * imp_mult).reshape(-1)
                idx = ((yy * W + xx).clamp(0, HW - 1) + boff).reshape(-1)
                color.index_add_(1, idx, src * wflat.unsqueeze(0))
                weight.index_add_(0, idx, wflat)
        out = (color / weight.clamp(min=1e-4)).reshape(C, M, H, W).permute(1, 0, 2, 3)
        holes = (weight < 1e-4).reshape(M, H, W)
        return out, holes

    def forward_flow_prev_grid(mv_c, depth_c):
        """
        mv_c   : (N,H,W,2) motion vectors of frame n (backward: pos_old = pos + mv), pixels, (x, y)
        depth_c: (N,H,W)   depth of frame n
        returns flow (N,2,H,W) = forward flow of frame n-1 (px per render step), holes (N,H,W) bool
        """
        # 1) the values to store: forward displacement is the negated mv
        val = (-mv_c).permute(0, 3, 1, 2).contiguous()           # (N,2,H,W)

        # 2) where each frame-n pixel was in frame n-1: p_old = p + mv
        dx, dy = mv_c[..., 0], mv_c[..., 1]                       # (N,H,W)

        # 3) if several pixels land on the same old pixel, the nearer surface wins
        imp = 1.0 / depth_c.clamp(min=0.1)                        # positive importance

        # 4) scatter val to p + mv, bilinearly, weighted by depth
        flow, holes = _splat_batch(val, dx, dy, imp, hw=1.0, beta=12.0)
        return flow, holes

    def fill_holes_nearest(flow, holes):
        """flow (2,H,W), holes (H,W) bool (torch) -> numpy (2,H,W), holes copied from the nearest valid pixel"""
        f = flow.cpu().numpy()
        h = holes.cpu().numpy()
        if h.any() and not h.all():
            idx = distance_transform_edt(h, return_distances=False, return_indices=True)  # row/col of nearest non-hole
            f = f[:, idx[0], idx[1]]
        return f

    def centroid(m):                       # m: (H,W) bool tensor on the GPU
        ys, xs = torch.nonzero(m, as_tuple=True)
        return torch.stack([xs.float().mean(), ys.float().mean()])
    
    def set_poses(t):
        observer = scene["observer"]
        target = scene["target"]
                    
        # --- 1. APPLY KINEMATIC TRAJECTORIES ---
        
        # Start from pristine default states
        obs_pose = observer.data.default_root_state[:, :7].clone()
        target_pose = target.data.default_root_state[:, :7].clone()

        # --- FLIGHT SETTINGS ---
        speed = 0.1 + 0.2 * math.cos(t)    # Reduced to 3.0 so they fly smoothly instead of teleporting
        radius = 4.0       # 4 meter wide circle
        delay = 0.5 + math.cos(t * speed * 2) / 4.0     # Observer follows 0.4 seconds behind

        # Calculate positions
        target_x = (radius + 0.2) * math.cos(t * speed )
        target_y = (radius + 0.2)* math.sin(t * speed)
        target_z = 2.0 + 0.15 * math.cos(t * speed * 2) # Bob up and down

        obs_x = radius * math.cos((t) * speed - delay)
        obs_y = radius * math.sin((t) * speed - delay)
        obs_z = 2.0

        # Assign positions
        target_pose[:, 0], target_pose[:, 1], target_pose[:, 2] = target_x, target_y, target_z
        obs_pose[:, 0], obs_pose[:, 1], obs_pose[:, 2] = obs_x, obs_y, obs_z

        # --- CALCULATE ROTATIONS (YAW) ---
        target_yaw = (t * speed) + (math.pi / 2.0)
        obs_yaw = ((t ) * speed - delay) + (math.pi / 2.0)

        # Apply Quaternions [qw, qx, qy, qz]
        target_pose[:, 3] = math.cos(target_yaw / 2.0)  
        target_pose[:, 4], target_pose[:, 5] = 0.0, 0.0 
        target_pose[:, 6] = math.sin(target_yaw / 2.0)  

        obs_pose[:, 3] = math.cos(obs_yaw / 2.0)
        obs_pose[:, 4], obs_pose[:, 5] = 0.0, 0.0
        obs_pose[:, 6] = math.sin(obs_yaw / 2.0)

        # Write poses to simulator
        target.write_root_pose_to_sim(target_pose)
        observer.write_root_pose_to_sim(obs_pose)

        # --- THE MISSING KILL SWITCH ---
        # You MUST zero the velocities here so the physics engine doesn't flip the drones!
        # zero_velocities = torch.zeros((1, 6), device=target.device)
        # target.write_root_velocity_to_sim(zero_velocities)
        # observer.write_root_velocity_to_sim(zero_velocities)
    
    # prev_snap = dvs.snapshot()
    # t_prev = 0.0
    import torch.nn.functional as F

    def mv_check(rgb_p, rgb_c, mv_c):
        N, H, W, _ = rgb_c.shape
        if mv_c.ndim == 3 and mv_c.shape[-1] != 2:
            mv_c = mv_c.permute(1, 2, 0)
        elif mv_c.ndim == 3 and mv_c.shape[0] == 2 and mv_c.shape[-1] == H:
            # Defensive: accept a legacy (2, H, W) field and convert to (H, W, 2)
            mv_c = mv_c.permute(1, 2, 0)

        ys, xs = torch.meshgrid(torch.arange(H, device=mv_c.device),
                                torch.arange(W, device=mv_c.device), indexing="ij")
        gx = (xs + mv_c[..., 0]) / (W - 1) * 2 - 1
        gy = (ys + mv_c[..., 1]) / (H - 1) * 2 - 1
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)  # (1, H, W, 2)
        grid = grid.expand(N, -1, -1, -1)
        rgb_p_chw = rgb_p.permute(0, 3, 1, 2)
        rgb_c_chw = rgb_c.permute(0, 3, 1, 2)
        warped = F.grid_sample(rgb_p_chw, grid, align_corners=True, padding_mode="border")
        return (warped - rgb_c_chw).abs().mean().item()


    # Create a folder for the 40Hz ground truth masks
    gt_dir = "/media/vboxuser/T9/sim_data/ground_truth"
    os.makedirs(gt_dir, exist_ok=True)

    # Find mapping of enemy drone

    enemy_id = None
    print("Starting simulation loop...")
    # t_prev = 0.0
    t_prev = -dt
    # In case the enemy drone does not start in a visible way, we will step the simulation until we find it so we can extract its ID
    while enemy_id is None and simulation_app.is_running():

        # 1. Step the simulation so the renderer actually draws the frame
        set_poses(t_prev) # Moves the drones by updating the buffers
        scene.write_data_to_sim() # Pushes buffers to PHYSX engine
        sim.step() # Advances sim by dt time. Runs the physics and then renders the frame
        scene.update(dt=dt) # Pulls the data from the sim and into the buffers where it can be accessed
        prev_snap = dvs.snapshot()
        t_prev += dt

        cam_data = scene["dvs_cam"].data
        if cam_data.info is not None and len(cam_data.info) > 0:
            info = cam_data.info[0]
            if "semantic_segmentation" in info:
                seg_info = info["semantic_segmentation"]
                if "idToLabels" in seg_info:
                    for obj_id_str, class_info in seg_info["idToLabels"].items():
                        if "enemy" in str(class_info):
                            enemy_id = ast.literal_eval(obj_id_str) # Converts string to tuple
                            print(f"[INFO] Found 'enemy' tag with Identifier: {enemy_id}")
                            break

    frame_count = 0
    tensor = None
    flow_list = []
    while simulation_app.is_running():
        try:
        
            set_poses(t_prev)
            # --- 2. STEP PHYSICS & RENDER (50Hz) ---
            scene.write_data_to_sim()
            sim.step()
            # simulation_app.update()
            scene.update(dt=dt)
            
            cam_data = scene["dvs_cam"].data
            

            # --- 3. DYNAMIC SEMANTIC ID LOOKUP ---
          
            # Change to whatever dt/40   
            if frame_count % 1 == 0:
                # --- 4. EXTRACT GROUND TRUTH SEGMENTATION ---
                seg_tensor = cam_data.output["semantic_segmentation"][0]
                
                # It's an RGBA color mask. We need to match all 4 channels (dim=-1)
                color_tensor = torch.tensor(enemy_id, device=seg_tensor.device, dtype=seg_tensor.dtype)
                # Check where the image pixels exactly match the RGBA color
                binary_mask = (seg_tensor == color_tensor).all(dim=-1).cpu().numpy().astype(np.uint8)
                if tensor is not None:
                    tensor_old = tensor
                tensor = (seg_tensor == color_tensor).all(dim=-1)

                vis_mask = binary_mask * 255 

                cv2.imshow("Ground Truth Mask", vis_mask)
                # Save the mask frame
                step_no = int(round((t_prev + dt) / dt))
                # np.save(f"{gt_dir}/mask_{step_no}.npy", binary_mask)

            # --- 5. SYNTHESIZE HIGH-FREQUENCY EVENTS (400Hz) ---
            cur_snap = dvs.snapshot()
            rgb_c, mv_c, depth_c = cur_snap["dvs_cam"]
            rgb_p = prev_snap["dvs_cam"][0]
            if frame_count >= 1:   # frame 0's mv points back to a warmup frame that has no saved mask
                flow_b, holes_b = forward_flow_prev_grid(mv_c, depth_c)
                flow_prev = fill_holes_nearest(flow_b[0], holes_b[0])      # (2,H,W) float32, forward flow of frame n-1
                flow_list.append(flow_prev.astype(np.float16))

            if frame_count % 25 == 1:
                # print(f"[diag] f={frame_count} mv_max={mv_c.abs().max().item():.3f}px "
                #     f"mv_nonzero={(mv_c.abs().sum(-1) > 0).float().mean().item():.4f} "
                #     f"rgb_change={(rgb_c - rgb_p).abs().mean().item():.6f}")
                
                # mv_c = np.squeeze(mv_c.cpu().numpy(), axis=0)  # Remove batch dimension
                # mv_u = mv_c[:, :, 0]  # Horizontal motion vector
                # mv_v = mv_c[:, :, 1]  # Vertical motion vector
                # flow_h, flow_w = mv_u.shape
                # step = max(1, min(flow_h, flow_w) // 24)
                # yy, xx = np.mgrid[0:flow_h:step, 0:flow_w:step]
                # quiver_u = mv_u[::step, ::step]
                # quiver_v = mv_v[::step, ::step]
                # mag = np.hypot(quiver_u, quiver_v)

                # fig, ax = plt.subplots(figsize=(8, 6))
                # ax.quiver(xx, yy, quiver_u, quiver_v,
                #         np.hypot(quiver_u, quiver_v),                       # color arrows by magnitude
                #         angles="xy", scale_units="xy", scale=1, cmap="plasma")
                # ax.set_xlim(0, flow_w)
                # ax.set_ylim(flow_h, 0)                                 # image coords: y down
                # ax.set_aspect("equal")
                # ax.axis("off")
                # plt.tight_layout()
                # # plt.show()
                # fig.savefig(f"{gt_dir}/flow/flow_{step_no}.png", dpi=150)
                # plt.close(fig)

                # in the loop, keep last step's target mask as tgt_prev (torch bool), current one is tgt_cur
                d = centroid(tensor) - centroid(tensor_old)        # how far the target moved, (x, y) px
                m = mv_c[0][tensor].median(dim=0).values         # typical mv on target pixels, (x, y)
                flow_t = torch.from_numpy(flow_list[-1].astype(np.float32)).to(tensor_old.device)   # (2,H,W)
                flow_t = flow_t.permute(1, 2, 0)  # (H,W,2)
                f = flow_t[tensor_old].median(dim=0).values
                # print("target moved", d.tolist(), "| mv on target", m.tolist(), " | new flow", f.tolist())

                print(mv_check(rgb_p, rgb_c, mv_c[0]), (rgb_p - rgb_c).abs().mean().item())
            dvs.warp_and_process(prev_snap, cur_snap, K, t_prev, dt_fine)
            
            # Advance time
            prev_snap = cur_snap
            t_prev += dt
            frame_count += 1
            cv2.waitKey(10)

            if frame_count >= 500:
                print("Reached 500 frames! Breaking loop...")
                break

        except Exception as e:
            # If ANY Python error happens, catch it and print it immediately!
            import traceback
            print("\n" + "="*50)
            print("🚨 CRASH DETECTED IN SIMULATION LOOP 🚨")
            print("="*50)
            traceback.print_exc()
            print("="*50 + "\n")
            break # Break the loop so it can shut down cleanly

    print("Flushing DVS events to disk...")
    try:
        dvs.flush(0, episode_idx=0)
        print("Flush command executed.")
        time.sleep(2) # Give the hard drive 2 seconds to write the massive file
    except Exception as e:
        print(f"FAILED TO FLUSH: {e}")    
    simulation_app.close()
if __name__ == "__main__":
    main()
    
