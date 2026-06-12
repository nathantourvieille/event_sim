import os
import carb
import math
# https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.utils/docs/index.html#isaacsim.core.utils.stage.create_new_stage
# https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.utils/docs/index.html#isaacsim.core.utils.stage.get_current_stage
from isaacsim.core.utils.stage import create_new_stage, get_current_stage, add_reference_to_stage
from isaacsim.core.utils.prims import create_prim
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

import carb

def configure_carb_settings(physics_dt=0.01, rendering_dt=0.01):
    print(f"[SHIM] Locking simulation time steps to dt = {physics_dt}s")
    settings = carb.settings.get_settings()
    
    # 1. Enforce fixed-step physics and rendering updates
    settings.set("/physics/maxStepSize", physics_dt)
    settings.set("/physics/minFrameRate", int(1.0 / physics_dt))
    settings.set("/physics/updateTicksPerFrame", 1)
    settings.set("/app/runLoops/main/rateLimitFrequency", int(1.0 / rendering_dt))
    settings.set("/omni/replicator/asyncRendering", False)

    # 2. Disable RTX Shadows & Ambient Occlusion (Performance)
    print("[SHIM] Disabling RTX Shadows and Ambient Occlusion...")
    settings.set("/rtx/shadows/enabled", False)
    settings.set("/rtx/ambientOcclusion/enabled", False)
    settings.set("/rtx/directLighting/sampledLighting/enabled", False) 

    # 3. Disable Cinematic Post-Processing (Data Integrity for Event Sim)
    print("[SHIM] Stripping cinematic post-processing to preserve sharp pixel gradients...")
    settings.set("/rtx/post/bloom/enabled", False)
    settings.set("/rtx/post/motionblur/enabled", False)
    settings.set("/rtx/post/depthOfField/enabled", False)
    settings.set("/rtx/post/lensFlares/enabled", False)
    settings.set("/rtx/post/chromaticAberration/enabled", False)

def setup_isaac_environment():

    print("[SHIM] Constructing USD Simulation Stage...")
    
    # Create an empty template stage
    create_new_stage()
    stage = get_current_stage()
    
    # 0. Load the Room Environment
    assets_root_path = get_assets_root_path()
    grid_usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
    add_reference_to_stage(usd_path=grid_usd_path, prim_path="/World/Room")

    # 1. Create a Default Light Source
    distant_light = create_prim(
        prim_path="/World/DistantLight",
        prim_type="DistantLight",
        position=(0.0, 0.0, 10.0)
    )
    distant_light.GetAttribute("inputs:intensity").Set(3000.0)
    
    # 2. Create the Sensor Camera
    # This path is what will be passed to rep.create.render_product()
    camera_path = "/World/TrackingCamera"
    camera_prim = create_prim(
        prim_path=camera_path,
        prim_type="Camera",
        position=(0.0, -5.0, 1.5), # Positioned back and slightly elevated
        orientation=(0.7071, 0.0, 0.0, 0.7071) # Pitched forward to look along Y axis
    )
    
    # 3. Create a Placeholder Target Object (e.g., a high-contrast sphere)
    # In production, this can be swapped with a custom drone USD asset path.
    target_path = "/World/TargetDrone"
    target_prim = create_prim(
        prim_path=target_path,
        prim_type="Sphere",
        position=(0.0, 5.0, 1.5), # Spawned 10 meters in front of the camera
    )
    # Scale down to proxy size of a small quadcopter (diameter = 30cm)
    target_prim.GetAttribute("xformOp:scale").Set((0.15, 0.15, 0.15))

    
    print(f"[SHIM] Environment ready. Tracking camera initialized at: {camera_path}")
    return camera_path


def update_camera_position(camera_path, frame_idx):
    """Move camera in circular orbit, like the drone in drone_test.py"""
    from pxr import Usd, UsdGeom
    
    stage = get_current_stage()
    camera_prim = stage.GetPrimAtPath(camera_path)
    
    # Simulated time variable
    t = frame_idx * 0.01
    
    # Circular orbit with vertical motion
    new_x = 8.0 * math.cos(t * 0.5)
    new_y = 8.0 * math.sin(t * 0.5)
    new_z = 2.0 + 0.5 * math.sin(t * 0.2)
    
    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    
    translate_op = xform.AddTranslateOp()
    translate_op.Set((new_x, new_y, new_z))



