import numpy as np
import cv2

def visualize_event_frame_live(bgr_out_wp):
    # 1. Pull the data from GPU back to the CPU
    bgr_image = bgr_out_wp.numpy() 
    
    # 2. THE CURE: Force standard memory alignment
    # We must explicitly cast to uint8, force it to be contiguous, 
    # and use .copy() to completely sever it from Warp's memory management.
    bgr_image = np.ascontiguousarray(bgr_image, dtype=np.uint8).copy()
    
    # 3. EMERGENCY BYPASS (Run this once to prove the data exists!)
    # If the window is still black, check your folder for this image file.
    cv2.imwrite("debug_green_test.png", bgr_image)

    # 4. Display
    cv2.imshow("Live Event Camera", bgr_image)
    key = cv2.waitKey(1)
    if key == 27:
        return False
    return True