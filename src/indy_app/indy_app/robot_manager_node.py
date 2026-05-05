import time, sys, os
import rclpy
# Đường dẫn môi trường ảo tuyệt đối để đảm bảo ổn định khi chạy ROS 2
VENV_PATH = os.path.expanduser("/home/apicoo-ai/pmg/bin_picking_ws/.venv/lib/python3.10/site-packages")
if os.path.exists(VENV_PATH) and VENV_PATH not in sys.path:
    sys.path.append(VENV_PATH)
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup,ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose, TransformStamped, PoseStamped
import tf2_geometry_msgs # Required for transform()
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from std_msgs.msg import String, Bool
import math
import numpy as np


ROBOT_IP = "192.168.1.135" # Cập nhật IP bot của bạn
DROP_POS = [38, -339, 300.0, 0, -179.99, 0.0] # Toạ độ thả (mm, x,y,z,u,v,w)
SWEEP_START = [38, -339, 300.0, 0, -179.99, 0.0]
SWEEP_END   = [38, -339, 300.0, 0, -179.99, 0.0]

def euler_to_quaternion(r, p, y): # rad
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    return [sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy]

def quaternion_to_euler(x, y, z, w):
    # Trả về Rx, Ry, Rz (Rad)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

class RobotManagerNode(Node):
    def __init__(self):
        super().__init__('robot_manager_node')
        
        # Xóa cấu hình use_sim_time mặc định để hệ thống đồng bộ với thời gian thực tế của dòng code driver ROS
        
        # ROS 2 TF Init
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(0.05, self.publish_robot_tf)
        self.create_timer(5, self.get_info)
        
    def publish_robot_tf(self):
        try:
            # Lấy toạ độ TCP hiện tại từ robot (mm, deg)
            p = self.tf_buffer.lookup_transform('link0', 'object_1', self.get_clock().now())
            print(p)

            # 1. Tạo thông điệp lệnh
            cmd = JointTrajectory()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'link0' # Quan trọng: Frame gốc của robot
            
            # Tên các khớp (Thường là 6 khớp của Indy)
            cmd.joint_names = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5']
            
            # 2. Tạo điểm đến (Target Point)
            point = JointTrajectoryPoint()
            
            # --- THAY THẾ PHẦN NÀY ---
            # Giả sử bạn đã tính được góc mục tiêu (deg) là: target_angles = [0, 0, 90, 0, 0, 0]
            # Hoặc bạn lấy từ Vision: target_angles = [vision_u, vision_v, vision_w, ...]
            target_angles = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0] 
            
            # Chuyển sang Rad và gán vào Point
            point.positions = [math.radians(a) for a in target_angles]
            
            # Thời gian di chuyển (Giây)
            point.time_from_start = self.get_clock().now() + Duration(sec=2)
            # --------------------------
            
            cmd.points.append(point)
            
            # 3. Gửi lệnh
            self.joint_trajectory_pub.publish(cmd)

        except: pass
        
    def get_info(self):
        try:
            # Dùng rclpy.time.Time() để lấy transform mới nhất thay vì dùng node clock the exact current time
            # Tên gốc tay máy Indy là 'link0' thay vì 'base_link' hay 'world'
            t = self.tf_buffer.lookup_transform('link0','link_object_0', rclpy.time.Time())
            self.get_logger().info(f"Target Object Position (rel. to link0): X={t.transform.translation.x:.3f}, Y={t.transform.translation.y:.3f}, Z={t.transform.translation.z:.3f}")
        except Exception as e:
            self.get_logger().warn(f"Waiting for TF: {e}", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = RobotManagerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
