import warp as wp

@wp.kernel
def compute_event_mask(
    current_frame: wp.array(dtype=wp.uint8, ndim=3),
    last_fired_frame: wp.array(dtype=wp.float32, ndim=2), # Intensity of the last event triggered
    previous_frame: wp.array(dtype=wp.float32, ndim=2), # Intensity of the last frame, used to find timestamps of events through linearizing
    bgr_out: wp.array(dtype=wp.uint8, ndim=3), # BGR image for debugging
    packed_events_1d: wp.array(dtype=wp.uint64, ndim=1), # Event list packed in mono byte format
    global_counter: wp.array(dtype=wp.int32, ndim=1), # Atomic counter, need an array if we want to be able to modify
    max_events: wp.int32,                               
    dt_frame_ns: wp.uint64,  # Time between current frame and the last
    threshold: float
):
    y, x = wp.tid()
   
    r = float(current_frame[y, x, 0])
    g = float(current_frame[y, x, 1])
    b = float(current_frame[y, x, 2])
    curr_val = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0 

    last_val = last_fired_frame[y, x]
    prev_val = previous_frame[y, x]
    diff = curr_val - last_val
    abs_diff = wp.abs(diff)
   
    if abs_diff >= threshold:
        event_count = wp.int32(wp.floor(abs_diff / threshold)) # Round down
        if diff > 0.0:
            direction = 1
        else:
            direction = -1
       
        if diff > 0.0:
            p_val = wp.uint64(1) # ON Event

            # ON Event: Write Red [B=0, G=0, R=255]
            bgr_out[y, x, 0] = wp.uint8(0)
            bgr_out[y, x, 1] = wp.uint8(0)
            bgr_out[y, x, 2] = wp.uint8(255)
            last_fired_frame[y, x] = last_val + (wp.float32(event_count) * threshold)  

        else:
            p_val = wp.uint64(0) # OFF Event

            # OFF Event: Write Blue [B=255, G=0, R=0]
            bgr_out[y, x, 0] = wp.uint8(255)
            bgr_out[y, x, 1] = wp.uint8(0)
            bgr_out[y, x, 2] = wp.uint8(0)
            last_fired_frame[y, x] = last_val - (wp.float32(event_count) * threshold)

        
        start_idx = wp.atomic_add(global_counter, 0, event_count) # Claim a block of indices in the packed_events_1d array
        available = max_events - start_idx # event_count can be modified during this time so we don't use it to calculate if we have overflowed

        if event_count <= available:

            x_idx = wp.uint64(x)
            y_idx = wp.uint64(y)

            # Need to find the timestamps for each event through linear interpolation
            for i in range(event_count):
                write_idx = start_idx + i

                frame_diff = curr_val - prev_val # Change in intensity between the two last frames
                if frame_diff == 0.0:
                    frame_diff = 0.001 # Prevent divide-by-zero

                # A. Calculate the exact intensity of the i'th event
                target_intensity = last_val + (wp.float32(i + 1) * threshold * wp.float32(direction))
                
                # B. Find where that intensity belongs along the slope from prev_val to curr_val
                fraction = (target_intensity - prev_val) / frame_diff
                
                # C. Clamp between 0.0 and 1.0 (Safety check to ensure time doesn't go backwards)
                fraction = wp.clamp(fraction, 0.0, 1.0)
                
                # D. Find the time at which event occured assuming the intensity changes linearly over dt_frame_ns
                curr_dt = wp.uint64(wp.float32(dt_frame_ns) * fraction)
                
                # E. Shift bits and pack event data into a single 64-bit integer mono format
                packed_event = (p_val << wp.uint64(63)) | (y_idx << wp.uint64(48)) | (x_idx << wp.uint64(32)) | curr_dt  

                packed_events_1d[write_idx] = packed_event
        else:
            # Overflow: undo the counter claim and do not write any events.
            wp.atomic_add(global_counter, 0, -event_count)
    else:
        # No Event: Standard Grey Background [B=127, G=127, R=127]
        bgr_out[y, x, 0] = wp.uint8(127)
        bgr_out[y, x, 1] = wp.uint8(127)
        bgr_out[y, x, 2] = wp.uint8(127)

    previous_frame[y, x] = curr_val # Event or no event we update the previous value

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
        self.prev_frame = wp.zeros((self.height, self.width), dtype=wp.float32, device=self.device)
        self.packed_events_1d = wp.zeros(self.max_events, dtype=wp.uint64, device=self.device)
        self.global_counter = wp.zeros(1, dtype=wp.int32, device=self.device)
        

    def process_frame(self, current_frame, dt_frame_ns):

        self.global_counter.zero_() # Set counter to zero at the start of each frame

        # Launch parallel execution grid across all pixels
        wp.launch(
            kernel=compute_event_mask,
            dim=(self.height, self.width),
            inputs=[current_frame, self.last_fired_frame, self.prev_frame, self.bgr_out, self.packed_events_1d, self.global_counter, self.max_events, dt_frame_ns, self.threshold],
            device=self.device
        )

        # Synchronization call if immediate synchronous access is required downstream
        wp.synchronize_device(self.device)
        
        count = int((self.global_counter.numpy())[0]) # Get the amount of events in CPU to be able to slice
        valid_events_cpu = self.packed_events_1d[:count].numpy() # Only grab what was used and move to CPU for ROS

        return valid_events_cpu