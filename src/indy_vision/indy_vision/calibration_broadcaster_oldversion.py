import os, sys
import json
import math
import numpy as np
import rclpy

# Đường dẫn môi trường ảo tuyệt đối để đảm bảo ổn định khi chạy ROS 2
VENV_PATH = os.path.expanduser("~/pmg/weld_vision_ws/.venv/lib/python3.10/site-packages")
if os.path.exists(VENV_PATH) and VENV_PATH not in sys.path:
    sys.path.append(VENV_PATH)

from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R

RESULT_FILE = "/home/apicoo-ai/pmg/bin_picking_ws/src/indy_vision/indy_vision/data_calib_intel/eye_in_hand_result.json"

def quaternion_from_matrix(m):
    # Sử dụng SciPy để đảm bảo độ chính xác tuyệt đối và đúng chuẩn ROS (x, y, z, w)
    r = R.from_matrix(m)
    return r.as_quat() # Trả về mảng [x, y, z, w] đúng chuẩn

class CalibrationBroadcaster(Node):
    def __init__(self):
        super().__init__('calibration_broadcaster')
        self.tf_bc = tf2_ros.StaticTransformBroadcaster(self)
        self.broadcast_tf()

    def broadcast_tf(self):
        now = self.get_clock().now().to_msg()
        transforms = []
        
        # 1. broadcast tcp -> link_tool0
        try:
            t1 = TransformStamped()
            t1.header.stamp = now
            t1.header.frame_id = 'tcp' 
            t1.child_frame_id = 'link_tool0'
            t1.transform.translation.x = 0.0
            t1.transform.translation.y = 0.0
            t1.transform.translation.z = 0.186 # 180mm
            t1.transform.rotation.w = 1.0
            transforms.append(t1)
        except Exception as e:
            self.get_logger().error(f"Error broadcasting tcp->link_tool0: {e}")

        # 2. broadcast link_tool0 -> link_camera
        if os.path.exists(RESULT_FILE):
            try:
                with open(RESULT_FILE, 'r') as f:
                    data = json.load(f)
                    T = np.array(data["T_cam_to_tool"])
                    print(T,flush=True)

                t2 = TransformStamped()
                t2.header.stamp = now
                t2.header.frame_id = 'link_tool0'  
                t2.child_frame_id = 'link_camera'

                q = quaternion_from_matrix(T[:3, :3])
                t2.transform.translation.x = T[0, 3]
                t2.transform.translation.y = T[1, 3]
                t2.transform.translation.z = T[2, 3]
                t2.transform.rotation.x = q[0]
                t2.transform.rotation.y = q[1]
                t2.transform.rotation.z = q[2]
                t2.transform.rotation.w = q[3]
                transforms.append(t2)
            except Exception as e:
                self.get_logger().error(f"Error broadcasting link_tool0->link_camera: {e}")

        if transforms:
            self.tf_bc.sendTransform(transforms)

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
