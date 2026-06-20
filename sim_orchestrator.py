import os
import carb
import math
from pxr import UsdGeom, Gf

# https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.utils/docs/index.html#isaacsim.core.utils.stage.create_new_stage
# https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.utils/docs/index.html#isaacsim.core.utils.stage.get_current_stage
from isaacsim.core.utils.stage import create_new_stage, get_current_stage, add_reference_to_stage
from isaacsim.core.utils.prims import create_prim
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf


def configure_carb_settings(sim_fps=100, max_fps=None):
    print(f"[SHIM] Locking simulation time steps to dt = {1 / sim_fps}s")
    settings = carb.settings.get_settings()
    
    # 1. Enforce fixed-step physics and rendering updates
    # If we know what the hardware can handle, we set that as the max so that the GPU is always safe
    if max_fps:
        settings.set("/app/runLoops/main/rateLimitEnabled",True)
        settings.set("/app/runLoops/main/rateLimitFrequency", max_fps) 

        settings.set("/time/timeScale", max_fps / sim_fps)  # Speed up the simulation time to maintain event density

    # settings.set("/physics/maxStepSize", 1 / sim_fps)
    # settings.set("/physics/minFrameRate", sim_fps)
    # settings.set("/physics/updateTicksPerFrame", 1)
    # settings.set("/physics/updateToCoreTime", False)
    settings.set("/app/asyncRendering", False)
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
    # grid_usd_path = assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd"
    grid_usd_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"

    add_reference_to_stage(usd_path=grid_usd_path, prim_path="/World/rivermark")

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


# def update_camera_position(camera_path, frame_idx):
#     """Move camera in circular orbit, like the drone in drone_test.py"""
#     from pxr import Usd, UsdGeom
    
#     stage = get_current_stage()
#     camera_prim = stage.GetPrimAtPath(camera_path)
    
#     # Simulated time variable
#     t = frame_idx * 0.01
    
#     # Circular orbit with vertical motion
#     new_x = 8.0 * math.cos(t * 0.5)
#     new_y = 8.0 * math.sin(t * 0.5)
#     new_z = 2.0 + 0.5 * math.sin(t * 0.2)
    
#     xform = UsdGeom.Xformable(camera_prim)
#     xform.ClearXformOpOrder()
    
#     translate_op = xform.AddTranslateOp()
#     translate_op.Set((new_x, new_y, new_z))




def get_drone_position(t):
    """Mathematical flight path of the drone at any given time 't'."""
    radius = 8.0
    
    # Flying in a circle (XY) while bobbing up and down (Z)
    x = radius * math.cos(t * 0.5)
    y = radius * math.sin(t * 0.5)
    z = 3.0 + 2.0 * math.sin(t)
    
    return Gf.Vec3d(x, y, z)

def update_camera_position(camera_path, frame_idx):
    """Moves camera along a path, facing forward in the direction of travel."""
    
    stage = get_current_stage()
    camera_prim = stage.GetPrimAtPath(camera_path)
    
    # 1. Current Time
    t = frame_idx * 0.01
    
    # 2. Calculate where we are NOW, and where we will be IN THE FUTURE
    current_pos = get_drone_position(t)
    future_pos = get_drone_position(t + 0.1) # Looking 0.1 simulated seconds ahead
    
    # 3. Calculate the FPV Look-At Matrix
    up_vector = Gf.Vec3d(0.0, 0.0, 1.0) # Z is up
    
    # By looking at our own future position, the nose of the drone points directly into the turn!
    view_matrix = Gf.Matrix4d().SetLookAt(current_pos, future_pos, up_vector)
    transform_matrix = view_matrix.GetInverse()
    
# 4. Apply the Transform Safely (THE FIX)
    xform = UsdGeom.Xformable(camera_prim)
    ops = xform.GetOrderedXformOps()
    
    # Check if we already have a clean Transform Matrix assigned.
    # If we don't (e.g., it's the very first frame), we wipe the slate and create one.
    if len(ops) != 1 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
        xform.ClearXformOpOrder()
        xform.AddTransformOp()
        ops = xform.GetOrderedXformOps() # Refresh our handle to the new operation
        
    # Now we just update the numbers inside the existing memory block.
    # This prevents the Vulkan renderer from crashing!
    ops[0].Set(transform_matrix)



