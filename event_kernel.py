from itertools import count

import warp as wp

@wp.kernel
def compute_event_mask(
    current_frame: wp.array(dtype=wp.uint8, ndim=3),
    last_fired_frame: wp.array(dtype=wp.float32, ndim=2),
    bgr_out: wp.array(dtype=wp.uint8, ndim=3), # Output a ready-to-draw BGR image directly
    packed_events_1d: wp.array(dtype=wp.uint64, ndim=1), # Event list packed in mono byte format
    global_counter: wp.array(dtype=wp.int32, ndim=1), # Atomic counter
    max_events: wp.int32,                                # Safeguard Limit
    dt_frame_ns: wp.uint64,  
    # events_out: wp.array(dtype=wp.int32, ndim=2),
    threshold: float
):
    y, x = wp.tid()
   
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
            p_val = wp.uint64(1) # ON Event

            bgr_out[y, x, 0] = wp.uint8(0)
            bgr_out[y, x, 1] = wp.uint8(0)
            bgr_out[y, x, 2] = wp.uint8(255)
            last_fired_frame[y, x] = last_val + (wp.float32(event_count) * threshold)
            # events_out[y, x] = event_count * direction
        else:
            p_val = wp.uint64(0) # OFF Event

            # OFF Event: Write Blue [B=255, G=0, R=0]
            bgr_out[y, x, 0] = wp.uint8(255)
            bgr_out[y, x, 1] = wp.uint8(0)
            bgr_out[y, x, 2] = wp.uint8(0)
            last_fired_frame[y, x] = last_val - (wp.float32(event_count) * threshold)
            # events_out[y, x] = event_count * direction

        
        start_idx = wp.atomic_add(global_counter, 0, event_count) # claim a block of indices in the packed_events_1d array
        if start_idx + event_count <= max_events:
            x_idx = wp.uint64(x)
            y_idx = wp.uint64(y)
            for i in range(event_count):
                write_idx = start_idx + i
                curr_dt = (wp.uint64(i) * dt_frame_ns) / wp.uint64(event_count) # Linearly interpolate timestamps for each event within the frame duration
                packed_event = (p_val << wp.uint64(63)) | (y_idx << wp.uint64(48)) | (x_idx << wp.uint64(32)) | curr_dt  # Shift bits and pack event data into a single 64-bit integer mono format
                packed_events_1d[write_idx] = packed_event
    else:
        # No Event: Standard Grey Background [B=127, G=127, R=127]
        bgr_out[y, x, 0] = wp.uint8(127)
        bgr_out[y, x, 1] = wp.uint8(127)
        bgr_out[y, x, 2] = wp.uint8(127)
        # events_out[y, x] = 0
       
class WarpEventCameraSimulator:
    def __init__(self, width=1920, height=1080, threshold=0.15):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.device = "cuda:0"
        self.max_events = 1_000_000

        # Pre-allocate static GPU memory grids once at startup
        self.bgr_out = wp.zeros((self.height, self.width, 3), dtype=wp.uint8, device=self.device)
        self.last_fired_frame = wp.zeros((self.height, self.width), dtype=wp.float32, device=self.device)
        self.packed_events_1d = wp.zeros(self.max_events, dtype=wp.uint64, device=self.device)
        self.global_counter = wp.zeros(1, dtype=wp.int32, device=self.device)
        
        
        # self.events_out = wp.zeros((self.height, self.width), dtype=wp.int32, device=self.device)

    def process_frame(self, current_frame, dt_frame_ns):

        self.global_counter.zero_() # Set counter to zero at the start of each frame

        # Launch parallel execution grid across all pixels
        wp.launch(
            kernel=compute_event_mask,
            dim=(self.height, self.width),
            inputs=[current_frame, self.last_fired_frame, self.bgr_out, self.packed_events_1d, self.global_counter, self.max_events, dt_frame_ns, self.threshold],
            device=self.device
        )

        # Synchronization call if immediate synchronous access is required downstream
        wp.synchronize_device(self.device)
        valid_events_cpu = self.packed_events_1d[:count].numpy()
        return valid_events_cpu