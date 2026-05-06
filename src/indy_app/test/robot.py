import time, sys, os
import rclpy
# Đường dẫn môi trường ảo tuyệt đối để đảm bảo ổn định khi chạy ROS 2
VENV_PATH = os.path.expanduser("/mnt/data/pmg/weld_vision_ws/.venv/lib/python3.10/site-packages")
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
import json
import threading
from neuromeka import IndyDCP3
from neuromeka.enums import OpState, TrajState, TaskBaseType, BlendingType,EndtoolState


ROBOT_IP = "192.168.1.2" # Cập nhật IP bot của bạn

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

def rad2deg(rx,ry,rz    ):
    return rx * 180 / math.pi,ry * 180 / math.pi,rz * 180 / math.pi 

class RobotManagerNode(Node):
    def __init__(self):
        super().__init__('robot_manager_node')
        self.robot= IndyDCP3(ROBOT_IP)
        self.robot_state = "waiting"
        self.vel_ratio = 20
        self.robot.set_speed_ratio(self.vel_ratio)
        self.target = None
        print("listen")


        # ROS 2 TF Init
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.state_pub=self.create_publisher(String,'/robot/state', 10)
        self.pos_sub = self.create_subscription(String,'/vision/pick_task', self.callback, 10)
        self.state_pub.publish(String(data=self.robot_state))
        # BẮT BUỘC: Phải có Timer để ROS gọi hàm run()
        self.create_timer(0.1, self.run)

    def callback(self, msg):
        if self.robot_state == "moving":
            print("robot is moving")
            return
        # KHÓA NGAY LẬP TỨC
        self.robot_state = "moving"
        self.state_pub.publish(String(data=self.robot_state))
        data = json.loads(msg.data)   # Chuyển string JSON → dict Python
        name = data["name"]
        pos  = data["pos"]
        quat = data["quat"]
        dims = data["dims"]
        try:
            # PHẢI có tiền tố 'target_'
            tf_name = name
            t = self.tf_buffer.lookup_transform('world', tf_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            x,y,z,qx,qy,qz,qw=t.transform.translation.x,t.transform.translation.y,t.transform.translation.z,t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w
            x,y,z = 1000*x,1000*y,1000*z+50
            rx,ry,rz=quaternion_to_euler(qx,qy,qz,qw)
            rx,ry,rz=rad2deg(rx,ry,rz)
            self.target = [x,y,z,rx,ry,rz]
            print(f"Target locked: {self.target} \n", flush=True)

        except Exception as e:
            print("TF not found")
            pass
        finally:
            self.robot.stop_motion()
            
    def run(self):
        try:
            if self.robot_state == "moving":
                time.sleep(5)    
                self.robot_state="waiting"
                self.state_pub.publish(String(data=self.robot_state))

        except Exception as e:
            print(e)
            pass

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
