import rclpy
from rclpy.node import Node
from event_camera_msgs.msg import EventArray
import numpy as np

class EventPublisherNode(Node):
    def __init__(self):
        super().__init__('isaac_event_camera_node')
        self.publisher_ = self.create_publisher(EventArray, '/event_camera/events', 10)

    def publish_events(self, event_byte_array, sec, nanosec, width, height):
        msg = EventArray()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = "camera_link"
        
        msg.width = width
        msg.height = height
        msg.is_bigendian = False 
        
        # You will need to specify the exact encoding your byte array uses
        # e.g., 'bif' (Binary Image Format) or a custom string defined by your packing logic
        msg.encoding = "custom_byte_encoding" 
        
        # Bring data from GPU (Warp) to CPU (Host) for standard ROS publishing
        msg.events = event_byte_array.numpy().tobytes()
        
        self.publisher_.publish(msg)