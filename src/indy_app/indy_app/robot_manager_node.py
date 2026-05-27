from numpy import rad2deg
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
from neuromeka import IndyDCP3,OpState

ROBOT_IP = "192.168.1.8" # Cập nhật IP bot của bạn
def euler_to_quaternion(r, p, y): # rad
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    return [sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy]

def quaternion_to_euler(x, y, z, w):
    # 1. Tính toán theo công thức chuẩn (nghiệm ZYX)
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
    
    # 2. Tìm nghiệm tương đương có pitch (Ry) gần -180 độ (tức là -pi rad)
    # Vì robot Indy luôn có tay gắp hướng xuống, Ry ở tư thế home là -180 độ
    roll_alt = roll_x - math.pi if roll_x > 0 else roll_x + math.pi
    if pitch_y > 0:
        pitch_alt = math.pi - pitch_y
    else:
        pitch_alt = -math.pi - pitch_y
    yaw_alt = yaw_z - math.pi if yaw_z > 0 else yaw_z + math.pi
    
    # Chuẩn hóa yaw_alt (Rz) về khoảng [-pi/2, pi/2] cho tay gắp đối xứng hai ngón susgrip
    # (Nếu xoay rz lệch 180 độ thì vẫn gắp giống nhau, giúp Joint 6 không bị xoay quá giới hạn)
    yaw_alt_deg = yaw_alt * 180.0 / math.pi
    while yaw_alt_deg > 90.0:
        yaw_alt_deg -= 180.0
    while yaw_alt_deg < -90.0:
        yaw_alt_deg += 180.0
    yaw_alt = yaw_alt_deg * math.pi / 180.0
    
    # Chọn bộ góc có pitch gần -pi (-180 độ) nhất
    if abs(pitch_alt - (-math.pi)) < abs(pitch_y - (-math.pi)):
        return roll_alt, pitch_alt, yaw_alt
    else:
        # Chuẩn hóa cả yaw_z cho nghiệm gốc nếu chọn
        yaw_z_deg = yaw_z * 180.0 / math.pi
        while yaw_z_deg > 90.0:
            yaw_z_deg -= 180.0
        while yaw_z_deg < -90.0:
            yaw_z_deg += 180.0
        yaw_z = yaw_z_deg * math.pi / 180.0
        return roll_x, pitch_y, yaw_z

def rad2deg(rx,ry,rz):
    return rx*180/math.pi,ry*180/math.pi,rz*180/math.pi
class RobotManagerNode(Node):
    def __init__(self):
        super().__init__('robot_manager_node')
        self.gripper = susgrip()
        # Xóa cấu hình use_sim_time mặc định để hệ thống đồng bộ với thời gian thực tế của dòng code driver ROS
        self.robot = IndyDCP3(ROBOT_IP, 0)
        # ROS 2 TF Init
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.sub = self.create_subscription(String, '/vision/pick_task', self.vision_callback, 10)
        self.pubstate = self.create_publisher(String, '/robot/state', 10)
        self.robotstate = "idle"
        self.pubstate.publish(String(data=self.robotstate))
        self.homepose = [350,-200,300,0 ,-180, 0] 
        self.placepose = [350,100,150,0 ,-180, 0] 
        self.startpose = [350,-365,300,0 ,-180, 0] 



    def vision_callback(self, msg):
        import json
        import rclpy.time
        from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
        if self.robotstate == "moving":
            return 
        self.robotstate = "moving"
        self.pubstate.publish(String(data=self.robotstate))
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
            Rx, Ry, Rz = quaternion_to_euler(
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            )
            self.get_logger().info(f"Đã tra cứu TF cho mục tiêu: {target_frame} | Pos: [{tf_x:.3f}, {tf_y:.3f}, {tf_z:.3f}]")
            
            # Gọi pick_and_place với tọa độ từ TF
            self.pick_and_place(tf_x, tf_y, tf_z, Rx, Ry, Rz)

        except (LookupException, ConnectivityException, ExtrapolationException) as ex:
            self.get_logger().error(f"Lỗi khi tra cứu TF cho mục tiêu '{target_frame}': {ex}")
        except Exception as e:
            self.get_logger().error(f"Lỗi xử lý JSON hoặc TF: {e}")

    def pick_and_place(self, x, y, z, rx, ry, rz):

        self.robot.movel(self.startpose, vel_ratio=10)
        self.robot.wait_for_operation_state(wait_op_state=OpState.IDLE)
        
        rx,ry,rz =rad2deg(rx,ry,rz)
        x,y,z = x*1000, y*1000, z*1000-15
        if z <=10: z=10
        robot_pose = [x, y, z, rx, ry, rz]
        print(robot_pose,flush=True)
        self.gripper.send_modbus_rtu_frame(100)

        self.robot.movel(robot_pose,vel_ratio=10)
        self.robot.wait_for_operation_state(wait_op_state=OpState.IDLE)

        self.gripper.send_modbus_rtu_frame(30)
        time.sleep(2)

        current = np.array(self.robot.get_robot_data()['p'])
        current[2] = 250
        print(current,flush =True)

        self.robot.movel(current,vel_ratio=10)
        self.robot.wait_for_operation_state(wait_op_state=OpState.IDLE)

        self.robot.movel(self.startpose,vel_ratio=10)
        self.robot.wait_for_operation_state(wait_op_state=OpState.IDLE)

        self.robot.movel(self.placepose,vel_ratio=10)
        self.robot.wait_for_operation_state(wait_op_state=OpState.IDLE)
        self.gripper.send_modbus_rtu_frame(100)

        self.robot.movel(self.startpose,vel_ratio=10)
        self.robot.wait_for_operation_state(wait_op_state=OpState.IDLE)


        self.robotstate = "idle"
        self.pubstate.publish(String(data=self.robotstate))

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
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
