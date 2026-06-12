import sys
import cv2
import gc
import torch
import time
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
from event_kernel import WarpEventCameraSimulator
from visualizer import visualize_event_frame_live

def main():
    print("[INIT] Booting Standalone Event Simulation Pipeline...")

    # 1. Lock the physics and rendering steps (e.g., dt = 0.01s / 100Hz)
    configure_carb_settings(physics_dt=0.02, rendering_dt=0.02)
   
    # 2. Build the world and return the camera prim path
    # (This hides all the messy USD and lighting setup)
    camera_path = setup_isaac_environment()
   
    # 3. Instantiate the Warp-based event simulator
    event_sim = WarpEventCameraSimulator(width=640, height=360, threshold=0.10)
   
    # 4. Create the Render Product and attach the Annotator
    render_product = rep.create.render_product(camera_path, resolution=(640, 360))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda")
    rgb_annotator.attach([render_product])
   
    print("[INIT] Pipeline established. Entering synchronous execution loop.")
   
    frame_idx = 0

    cv2.namedWindow("Live Event Camera", cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread() # <--- THIS IS THE CRITICAL FIX

    print("OpenCV Window Thread Started. Initializing Isaac Sim...")
   
    # 5. The Ghost Kitchen Loop (Dedicated thread, maximum execution speed)
    while simulation_app.is_running():
        try:
            # A. You drive the engine. This executes your 0.01s physics tick and renders the scene.
            simulation_app.update()
            
            # A1. Move the camera (simple)
            update_camera_position(camera_path, frame_idx)
           
            # B. Fire Replicator to harvest the data for THIS exact state.
            # https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/latest/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.orchestrator.step
            # delta_time=0.0 ensures Replicator grabs the buffer without double-stepping the physics timeline.
            # wait_for_render=True ensures the Python execution thread blokcs until the GPU is done
            # pause_timeline=True ensures the Omniverse timeline clock is frozen
            # rt_subframes=-1 ensures it uses the modified carb settings
            rep.orchestrator.step(delta_time=0.0, wait_for_render=True, pause_timeline=True, rt_subframes=-1)
           
            # C. Retrieve the dense array directly from the VRAM buffer
            # https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/latest/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.annotators.Annotator.get_data
            # device=cuda ensures a warp array is returned and remains in VRAM
            # do_array_copy ensures a copy of the frame is made in case the events_out_wp is not computed in time
            rgb_data = rgb_annotator.get_data(device="cuda", do_array_copy=False)
           

            # D. Route to your custom NVIDIA Warp kernel
            events_out_wp = event_sim.process_frame(rgb_data)
           
            # E. Live validation
            keep_running = visualize_event_frame_live(events_out_wp)
           
            # if not keep_running:
            #     print("\n[STOP] User interrupted via OpenCV window. Exiting loop.")
            #     break

            time.sleep(0.2) 
            frame_idx += 1
           
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")
                # gc.collect()
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\n[STOP] Keyboard interrupt detected. Exiting loop.")
            break
       
    # 6. Clean up windows before closing
    cv2.destroyAllWindows()
    rep.orchestrator.stop()
    import omni.timeline
    omni.timeline.get_timeline_interface().stop()
    from omni.isaac.core.world import World
    World.clear_instance()
    simulation_app.close()

    print("[CLEANUP] Simulation terminated successfully.")

if __name__ == "__main__":
    main()