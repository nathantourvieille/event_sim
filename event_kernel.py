import warp as wp

@wp.kernel
def test_paint_green(bgr_out: wp.array(dtype=wp.uint8, ndim=3)):
    y, x = wp.tid()
    bgr_out[y, x, 0] = wp.uint8(0)   # Blue
    bgr_out[y, x, 1] = wp.uint8(255) # Green
    bgr_out[y, x, 2] = wp.uint8(0)   # Red

@wp.kernel
def compute_event_mask(
    current_frame: wp.array(dtype=wp.uint8, ndim=3),
    last_fired_frame: wp.array(dtype=wp.float32, ndim=2),
    bgr_out: wp.array(dtype=wp.uint8, ndim=3), # Output a ready-to-draw BGR image directly
    events_out: wp.array(dtype=wp.int32, ndim=2),
    threshold: float
):
    y, x = wp.tid()
    
    # Bounds check
    if y >= current_frame.shape[0] or x >= current_frame.shape[1]:
        return
   
    r = float(current_frame[y, x, 0])
    g = float(current_frame[y, x, 1])
    b = float(current_frame[y, x, 2])
    cur_val = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0 

    last_val = last_fired_frame[y, x]
    diff = cur_val - last_val
    abs_diff = wp.abs(diff)
   
    if abs_diff >= threshold:
        event_count = wp.int32(wp.floor(abs_diff / threshold))
        if diff > 0.0:
            direction = 1
        else:
            direction = -1
       
        if diff > 0.0:
            # ON Event: Write Red [B=0, G=0, R=255]
            bgr_out[y, x, 0] = wp.uint8(0)
            bgr_out[y, x, 1] = wp.uint8(0)
            bgr_out[y, x, 2] = wp.uint8(255)
            last_fired_frame[y, x] = last_val + (wp.float32(event_count) * threshold)
            events_out[y, x] = event_count * direction
        else:
            # OFF Event: Write Blue [B=255, G=0, R=0]
            bgr_out[y, x, 0] = wp.uint8(255)
            bgr_out[y, x, 1] = wp.uint8(0)
            bgr_out[y, x, 2] = wp.uint8(0)
            last_fired_frame[y, x] = last_val - (wp.float32(event_count) * threshold)
            events_out[y, x] = event_count * direction
    else:
        # No Event: Standard Grey Background [B=127, G=127, R=127]
        bgr_out[y, x, 0] = wp.uint8(127)
        bgr_out[y, x, 1] = wp.uint8(127)
        bgr_out[y, x, 2] = wp.uint8(127)
        events_out[y, x] = 0
       
class WarpEventCameraSimulator:
    def __init__(self, width=1920, height=1080, threshold=0.15):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.device = "cuda:0"
       
        # Pre-allocate static GPU memory grids once at startup
        self.bgr_out = wp.zeros((self.height, self.width, 3), dtype=wp.uint8, device=self.device)
        self.last_fired_frame = wp.zeros((self.height, self.width), dtype=wp.float32, device=self.device)
        self.events_out = wp.zeros((self.height, self.width), dtype=wp.int32, device=self.device)

    def process_frame(self, current_frame):


        # Launch parallel execution grid across all pixels
        wp.launch(
            kernel=compute_event_mask,
            dim=(self.height, self.width),
            inputs=[current_frame, self.last_fired_frame, self.bgr_out, self.events_out, self.threshold],
            device=self.device
        )

        # Synchronization call if immediate synchronous access is required downstream
        wp.synchronize_device(self.device)
       
        return self.bgr_out