# /home/apicoo-ai/pmg/bin_picking_ws/viz_avoidance.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import threading
import queue
from tf2_ros import TransformListener, Buffer

class AvoidanceViz(Node):
    def __init__(self, traj_queue):
        super().__init__('avoidance_viz')
        self.traj_queue = traj_queue
        from rclpy.callback_groups import ReentrantCallbackGroup
        self.group = ReentrantCallbackGroup()

        # TF Listener để dịch tọa độ từ Camera sang World
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pcl_sub = self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.pcl_callback, 10, callback_group=self.group)
        self.traj_sub = self.create_subscription(DisplayTrajectory, '/display_planned_path', self.traj_callback, 10, callback_group=self.group)
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk', callback_group=self.group)
        
        self.latest_pcd = None
        print("Đang đợi Camera và lệnh 'Plan' từ RViz...")

    def pcl_callback(self, msg):
        try:
            # Tìm phép biến đổi từ Camera sang World tại thời điểm hiện tại
            trans = self.tf_buffer.lookup_transform('world', msg.header.frame_id, rclpy.time.Time())
            
            # Đọc điểm và áp dụng phép biến đổi tọa độ
            points_gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
            points = np.array([list(p) for p in points_gen], dtype=np.float32)
            
            if points.size == 0: return

            # Chuyển đổi tọa độ dùng ma trận (Quay và Dịch)
            from scipy.spatial.transform import Rotation as R
            quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            rot = R.from_quat(quat).as_matrix()
            offset = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            
            # Tọa độ mới = (Xoay * Tọa độ cũ) + Dịch chuyển
            points_world = (points @ rot.T) + offset
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            self.latest_pcd = pcd.voxel_down_sample(voxel_size=0.01)
        except Exception as e:
            self.get_logger().warn(f"Chưa có TF: {e}")

    async def traj_callback(self, msg):
        if self.latest_pcd is None:
            print("Chưa nhận được PCL hoặc TF!")
            return

        print("\n=== Đang dựng phối cảnh 3D (Đã khớp tọa độ World)... ===")
        points = msg.trajectory[0].joint_trajectory.points
        joint_names = msg.trajectory[0].joint_trajectory.joint_names
        
        tcp_path = []
        indices = np.linspace(0, len(points) - 1, 10, dtype=int)
        for i in indices:
            pt = points[i]
            req = GetPositionFK.Request()
            req.header.frame_id = "world"
            req.fk_link_names = ["tcp"]
            req.robot_state.joint_state = JointState(name=joint_names, position=pt.positions)
            
            future = self.fk_client.call_async(req)
            res = await future
            if res and res.pose_stamped:
                p = res.pose_stamped[0].pose.position
                tcp_path.append([p.x, p.y, p.z])

        if tcp_path:
            self.traj_queue.put((self.latest_pcd, tcp_path))

def show_3d_viz(pcd, path):
    # (Đoạn này giữ nguyên như cũ để vẽ)
    line_set = o3d.geometry.LineSet()
    pts = np.array(path)
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
    spheres = []
    for p in pts:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        s.paint_uniform_color([0, 1, 0]); s.translate(p)
        spheres.append(s)
    o3d.visualization.draw_geometries([pcd, line_set] + spheres)

def main():
    rclpy.init()
    traj_queue = queue.Queue()
    node = AvoidanceViz(traj_queue)
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        while rclpy.ok():
            try:
                pcd, path = traj_queue.get(timeout=0.1)
                show_3d_viz(pcd, path)
            except: continue
    except KeyboardInterrupt: pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
