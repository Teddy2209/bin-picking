import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
import math

class SimpleRobotManager(Node):
    def __init__(self):
        super().__init__('simple_robot_manager')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.get_logger().info('Đang chờ MoveIt Action Server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Đã kết nối với MoveIt!')

    def send_goal(self, x, y, z, roll, pitch, yaw):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'indy_manipulator'
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.2 # Chạy chậm để dễ quan sát
        goal_msg.request.max_acceleration_scaling_factor = 0.2
        
        # 1. Tạo Pose mục tiêu
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'link0'
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        
        q = self.euler_to_quaternion(roll, pitch, yaw)
        target_pose.pose.orientation.x = q[0]
        target_pose.pose.orientation.y = q[1]
        target_pose.pose.orientation.z = q[2]
        target_pose.pose.orientation.w = q[3]

        # 2. Đóng gói vào Constraints (Phần này nãy bị thiếu)
        constraints = Constraints()
        constraints.name = "goal_constraints"
        
        # Ràng buộc vị trí (Position Constraint)
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = 'link0'
        pos_constraint.link_name = 'tcp' # Điểm sẽ tới đích là TCP
        
        # Tạo một vùng nhỏ quanh điểm đích (sai số 1mm)
        bv = BoundingVolume()
        s = SolidPrimitive()
        s.type = SolidPrimitive.SPHERE
        s.dimensions = [0.001] # Bán kính 1mm
        bv.primitives.append(s)
        bv.primitive_poses.append(target_pose.pose)
        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0
        
        # Ràng buộc hướng (Orientation Constraint)
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = 'link0'
        ori_constraint.link_name = 'tcp'
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.01
        ori_constraint.absolute_y_axis_tolerance = 0.01
        ori_constraint.absolute_z_axis_tolerance = 0.01
        ori_constraint.weight = 1.0
        
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(ori_constraint)
        
        goal_msg.request.goal_constraints.append(constraints)
        
        # 3. Gửi lệnh
        self.get_logger().info(f'Đang gửi yêu cầu di chuyển tới: X={x}, Y={y}, Z={z}')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('MoveIt từ chối mục tiêu! (Có thể robot không tới được vị trí này)')
            return
        self.get_logger().info('MoveIt đã chấp nhận lệnh, đang lập kế hoạch và di chuyển...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        status = future.result().status
        # GoalStatus.STATUS_SUCCEEDED = 4 trong ROS 2
        if status == 4:
            self.get_logger().info('Robot đã đến đích thành công! Rút lui thôi!')
        else:
            self.get_logger().warn(f'Di chuyển kết thúc với trạng thái: {status}')

    def euler_to_quaternion(self, r, p, y):
        (r, p, y) = (math.radians(r), math.radians(p), math.radians(y))
        cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
        cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
        cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
        return [sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy]

def main(args=None):
    rclpy.init(args=args)
    manager = SimpleRobotManager()
    
    # Thử một tọa độ an toàn: X=0.4m, Y=0.2m, Z=0.4m (Tầm với tay Indy7 khoảng 0.7m)
    # Lưu ý: Z phải dương (Robot nằm trên mặt đất)
    manager.send_goal(0.4, 0.2, 0.4, 0.0, -180.0, 0.0)
    
    rclpy.spin(manager)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
