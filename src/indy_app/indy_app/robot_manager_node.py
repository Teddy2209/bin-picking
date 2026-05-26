import time, sys, os
import rclpy
# Đường dẫn môi trường ảo tuyệt đối để đảm bảo ổn định khi chạy ROS 2
VENV_PATH = os.path.expanduser("/home/apicoo-ai/pmg/bin_picking_ws/.venv/lib/python3.10/site-packages")
if os.path.exists(VENV_PATH) and VENV_PATH not in sys.path:
    sys.path.append(VENV_PATH)

# Đường dẫn workspace để có thể import các file code nằm ở thư mục gốc như modbus_test.py
WORKSPACE_PATH = "/home/apicoo-ai/pmg/bin_picking_ws"
if os.path.exists(WORKSPACE_PATH) and WORKSPACE_PATH not in sys.path:
    sys.path.append(WORKSPACE_PATH)

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
from modbus_test import susgrip


ROBOT_IP = "192.168.1.135" # Cập nhật IP bot của bạn


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
        self.gripper = susgrip()
        # Xóa cấu hình use_sim_time mặc định để hệ thống đồng bộ với thời gian thực tế của dòng code driver ROS
        
        # ROS 2 TF Init
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.sub = self.create_subscription(String, '/vision/pick_task', self.vision_callback, 10)

        self.homepose = [287, -333, 292, 0, -180, 0]
        self.placepose = [0, 0, 0, 0, 0, 0] # Thay thế bằng tọa độ đặt của bạn

    def vision_callback(self, msg):
        import json
        import rclpy.time
        from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
        
        try:
            # Lấy tên vật thể từ dữ liệu JSON do vision_node đẩy lên
            task_data = json.loads(msg.data)
            c_name = task_data.get('name')
            
            # Khởi tạo target_frame giống với lúc được broadcast ở vision_node
            target_frame = f"target_{c_name}"
            # Lấy TF tới base của robot (có thể đổi thành 'link0' nếu cấu hình của bạn là link0)
            base_frame = "world"
            
            # Tra cứu toạ độ hiện tại từ TF
            t = self.tf_buffer.lookup_transform(
                base_frame,
                target_frame,
                rclpy.time.Time()
            )
            
            # Lấy tọa độ x, y, z
            tf_x = t.transform.translation.x
            tf_y = t.transform.translation.y
            tf_z = t.transform.translation.z
            
            self.get_logger().info(f"Đã tra cứu TF cho mục tiêu: {target_frame} | Pos: [{tf_x:.3f}, {tf_y:.3f}, {tf_z:.3f}]")
            
            # Gọi pick_and_place với tọa độ từ TF
            self.pick_and_place(tf_x, tf_y, tf_z)
                
        except (LookupException, ConnectivityException, ExtrapolationException) as ex:
            self.get_logger().error(f"Lỗi khi tra cứu TF cho mục tiêu '{target_frame}': {ex}")
        except Exception as e:
            self.get_logger().error(f"Lỗi xử lý JSON hoặc TF: {e}")

    def pick_and_place(self, x, y, z):
        self.get_logger().info(f"Thực hiện Pick & Place tại: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        print("x = ", x)
        print("y = ", y)
        print("z = ", z)
        # Các lệnh movel() của bạn để đi tới điểm gắp
        # movel()


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
