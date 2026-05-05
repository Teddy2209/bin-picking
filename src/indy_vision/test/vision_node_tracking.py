"""
Vision Node (ROS 2) - Intel RealSense + Ultralytics SAM
- Classes: AIEngine, VisionApp, VisionNode
- Workflow: Click -> segment -> '+' to add -> 'Enter' to lock list and continuously track/broadcast.
"""
import os, sys, json, threading, math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R_sci
import torch

# Cutie imports
sys.path.append('/home/apicoo-ai/pmg/weld_vision_ws/Cutie')
try:
    from cutie.utils.get_default_model import get_default_model
    from cutie.inference.inference_core import InferenceCore
except ImportError:
    pass

def image_to_torch(frame: np.ndarray, device: str = 'cuda'):
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
    return frame

# ============================================================
# CẤU HÌNH
# ============================================================
COLOR_W    = 1280
COLOR_H    = 720
FPS        = 30
DEPTH_MIN  = 0.1
DEPTH_MAX  = 2.0
WIN_NAME   = "Vision Node"
INTRINSIC_FILE = "/home/apicoo-ai/pmg/weld_vision_ws/weld_core/data_calib_intel/camera_intrinsics.json"
MODEL_PATH = '/home/apicoo-ai/pmg/weld_vision_ws/models/mobile_sam.pt'

# ============================================================
# ROS 2 NODE (TF Broadcaster)
# ============================================================
class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info("VisionNode ready. Broadcasting to TF2...")

    def broadcast_tf(self, name, x, y, z, qx, qy, qz, qw):
        # Liên tục gửi tọa độ và hướng quaternion (Qx, Qy, Qz, Qw) lên TF2
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "link_camera"
        t.child_frame_id = f"obj_{name}"
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(t)

# ============================================================
# AI ENGINE
# ============================================================
class AIEngine:
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        # Đẩy quá trình tải model sang Thread ngầm để không cản trở việc mở Camera
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        self.logger.info("Đang Load Model AI ở background...")
        try:
            from ultralytics import SAM
            # Load SAM. Bạn có thể thay đổi device='cuda' hoặc 'cpu' trong model()
            self.model = SAM(MODEL_PATH)
            self.logger.info(f"Đã load xong AI ({MODEL_PATH})!")
        except Exception as e:
            self.logger.error(f"Lỗi khởi tạo AI: {e}")

    def get_mask_contour(self, image, px, py):
        if self.model is None: return None
        try:
            # Chạy Inference cực nhanh
            res = self.model(image, points=[[px, py]], labels=[1], device='cuda', verbose=False)
            if res and res[0].masks is not None:
                contours = res[0].masks.xy
                if len(contours) > 0:
                    return np.array(contours[0], dtype=np.int32)
        except Exception as e:
            self.logger.error(f"AI Error: {e}")
        return None

# ============================================================
# OBJECT TRACKER (Kalman filter + similarity check)
# ============================================================
class ObjectTracker:
    def __init__(self, obj_id, initial_center, initial_contour):
        self.id = obj_id
        self.last_center = np.array(initial_center, dtype=np.float32)
        self.last_contour = initial_contour
        self.area = cv2.contourArea(initial_contour)
        # Kalman filter (2D: x, y, vx, vy)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.statePre = np.array([[initial_center[0]], [initial_center[1]], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[initial_center[0]], [initial_center[1]], [0], [0]], np.float32)
        self.predicted_center = initial_center
        self.lost_counter = 0

    def predict(self):
        """Dự đoán tâm mới dựa trên Kalman"""
        pred = self.kalman.predict()
        self.predicted_center = (int(pred[0]), int(pred[1]))
        return self.predicted_center

    def update(self, measured_center, contour):
        """Cập nhật Kalman với measurement thực tế, kiểm tra tính hợp lệ"""
        new_area = cv2.contourArea(contour)
        area_ratio = new_area / self.area if self.area > 0 else 1.0
        # Nếu diện tích thay đổi quá 40% (có thể do chuyển vật), coi như mất dấu
        if area_ratio < 0.6 or area_ratio > 1.4:
            self.lost_counter += 1
            # Không cập nhật Kalman, chỉ dùng dự đoán
            return False
        else:
            self.lost_counter = 0
            self.area = new_area
            self.last_center = np.array(measured_center, dtype=np.float32)
            self.last_contour = contour
            # Cập nhật Kalman
            measurement = np.array([[measured_center[0]], [measured_center[1]]], np.float32)
            self.kalman.correct(measurement)
            return True

    def is_lost(self):
        return self.lost_counter > 5  # mất dấu sau 5 frame liên tiếp



# ============================================================
# VISON APP (GUI & Logic)
# ============================================================
class VisionApp:
    def __init__(self, node: VisionNode):
        self.node = node
        self.ai = AIEngine(self.node.get_logger())
        self.trackers = []  # list of ObjectTracker
        self.temp_selected = None 
        self.confirmed_objects = [] 
        self.is_tracking = False 
        self._custom_intrinsics = None # Fix lỗi AttributeError

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.node.get_logger().info("Đang load CUTIE Memory Model...")
        try:
            self.cutie_model = get_default_model()
            self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
        except Exception as e:
            self.node.get_logger().error(f"Lỗi load Cutie: {e}")

        # Khởi động RealSense ngay lập tức
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, COLOR_W, COLOR_H, rs.format.z16,  FPS)
        profile = self._pipeline.start(cfg)

        self._rs_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._active_intrinsics = self._rs_intrinsics
        self._load_custom_intrinsics()

        self._spatial = rs.spatial_filter()
        self._hole    = rs.hole_filling_filter()
        self._colmap  = rs.colorizer()
        self._align   = rs.align(rs.stream.color)

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 720)
        cv2.setMouseCallback(WIN_NAME, self._on_mouse)

        self._color_img = None
        self._depth_raw = None

    def _load_custom_intrinsics(self):
        if os.path.exists(INTRINSIC_FILE):
            try:
                with open(INTRINSIC_FILE) as f:
                    d = json.load(f)
                K = np.array(d["camera_matrix"])
                D = np.array(d["dist_coeffs"]).flatten()
                
                intr = rs.intrinsics()
                intr.width = d["image_size"][0]
                intr.height = d["image_size"][1]
                intr.ppx = K[0, 2]
                intr.ppy = K[1, 2]
                intr.fx = K[0, 0]
                intr.fy = K[1, 1]
                intr.model = rs.distortion.brown_conrady
                intr.coeffs = [D[0], D[1], D[2], D[3], D[4]]
                
                self._custom_intrinsics = intr
                self._active_intrinsics = intr
                self.node.get_logger().info("Đã load custom intrinsics.")
            except Exception as e:
                self.node.get_logger().warn(f"Failed custom intrinsics: {e}")

    def _on_mouse(self, event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN: return
        if self.is_tracking: return # Khóa chuột khi đang tracking
        if self._color_img is None: return

        contour = self.ai.get_mask_contour(self._color_img, mx, my)
        if contour is not None:
            self.temp_selected = {
                "contour": contour,
                "center_2d": (mx, my)
            }
            self.node.get_logger().info("Đã segment 1 vật. Nhấn '+' để LƯU, 'Enter' để BẮT ĐẦU TRACKING.")
        else:
            self.node.get_logger().warn("AI không tìm thấy vùng nào!")

    def _extract_3d_pose(self, contour):
        """
        Thuật toán Real-time: 
        Dùng numpy PCA trực tiếp trên mây điểm thay vì Open3D RANSAC, 
        đảm bảo tốc độ 30 khung hình/giây không gây giật lag Camera.
        """
        if self._depth_raw is None: return None
        
        # 1. Tạo mask 2D đen trắng từ contour đa giác
        mask_img = np.zeros((COLOR_H, COLOR_W), dtype=np.uint8)
        cv2.drawContours(mask_img, [contour], -1, 255, -1)
        ys, xs = np.where(mask_img == 255)
        
        # 2. Subsample giảm tải CPU
        step = max(1, len(xs) // 1000)
        xs, ys = xs[::step], ys[::step]
        
        # 3. Lấy tập hợp điểm 3D (Point Cloud nội bộ)
        intr = getattr(self, '_custom_intrinsics', None) or self._active_intrinsics
        pts_3d = []
        for u, v in zip(xs, ys):
            d = self._depth_raw.get_distance(int(u), int(v))
            if DEPTH_MIN < d < DEPTH_MAX:
                pt = rs.rs2_deproject_pixel_to_point(intr, [int(u), int(v)], d)
                pts_3d.append(pt)
                
        if len(pts_3d) < 10: return None
        
        pts_array = np.array(pts_3d)
        
        # 4. Lọc Outlier (Loại bỏ các điểm background bị lọt vào rìa Mask)
        z_values = pts_array[:, 2]
        z_median = np.median(z_values)
        # Chỉ lấy các điểm có độ sâu chênh lệch +- 3cm so với median
        valid_mask = np.abs(z_values - z_median) < 0.03 
        valid_pts = pts_array[valid_mask]
        
        if len(valid_pts) < 5: 
            valid_pts = pts_array # Nếu lọc gắt quá bị mất hết, thì dùng lại mảng cũ
            
        # Tâm hình học ổn định
        cx, cy, cz = np.mean(valid_pts, axis=0)
        
        # 5. PCA để tìm góc quay (Orientation) dùng valid_pts để đỡ nhiễu
        cov = np.cov(valid_pts, rowvar=False)
        val, vec = np.linalg.eigh(cov)
        
        # Sắp xếp theo mức độ ảnh hưởng của các trục
        idx = val.argsort()[::-1]
        vec = vec[:, idx]
        
        # Trục Z là trục ngắn nhất của vật (mặt bẹt)
        V_normal = vec[:, 2]
        if V_normal[2] < 0: V_normal = -V_normal # Luôn hướng xuống bàn (Z dương)
            
        # Trục X là trục dài nhất của vật
        V_major = vec[:, 0]
        
        grasp_Z = V_normal
        grasp_X = np.cross(V_major, grasp_Z)
        norm_X = np.linalg.norm(grasp_X)
        if norm_X == 0: return None
        grasp_X = grasp_X / norm_X
        if grasp_X[0] < 0: grasp_X = -grasp_X
            
        grasp_Y = np.cross(grasp_Z, grasp_X)
        
        R_cam_grasp = np.column_stack((grasp_X, grasp_Y, grasp_Z))
        quat = R_sci.from_matrix(R_cam_grasp).as_quat()
        print("cx, cy, cz, quat[0], quat[1], quat[2], quat[3]", cx, cy, cz, quat[0], quat[1], quat[2], quat[3])
        return cx, cy, cz, quat[0], quat[1], quat[2], quat[3]

    def run(self):
        try:
            while rclpy.ok():
                try: frames = self._pipeline.wait_for_frames(100)
                except Exception: frames = None
                
                if frames and frames.size() >= 2:
                    aligned = self._align.process(frames)
                    df, cf = aligned.get_depth_frame(), aligned.get_color_frame()
                    if df and cf:
                        df = self._spatial.process(df)
                        df = self._hole.process(df)
                        self._depth_raw = df.as_depth_frame()
                        self._color_img = np.asanyarray(cf.get_data())
                        
                if self._color_img is None: continue
                
                display = self._color_img.copy()
                
                # Chuyển frame sang format cho Cutie
                frame_rgb = cv2.cvtColor(self._color_img, cv2.COLOR_BGR2RGB)
                frame_tensor = image_to_torch(frame_rgb, device=self.device)

                # =======================================================
                # 🔄 CHẾ ĐỘ TRACKING VÀ PHÁT SÓNG TF2 LIÊN TỤC BẰNG CUTIE
                # =======================================================
                if self.is_tracking:
                    with torch.inference_mode():
                        pred_mask_tensor = self.processor.step(frame_tensor)
                        pred_mask_np = torch.max(pred_mask_tensor, dim=0).indices.detach().cpu().numpy().astype(np.uint8)

                    for i, obj in enumerate(self.confirmed_objects):
                        obj_id = i + 1
                        
                        if np.any(pred_mask_np == obj_id):
                            # Trích xuất contour mới từ Mask của Cutie
                            obj_mask_binary = (pred_mask_np == obj_id).astype(np.uint8)
                            contours, _ = cv2.findContours(obj_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                new_contour = max(contours, key=cv2.contourArea)
                                obj["contour"] = new_contour
                                
                                # Cập nhật tâm 2D
                                M = cv2.moments(new_contour)
                                if M["m00"] != 0:
                                    obj["center_2d"] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                            
                                # Vẽ mask và contour lên màn hình
                                color_mask = np.zeros_like(display)
                                color_mask[pred_mask_np == obj_id] = [0, 255, 0]
                                display = cv2.addWeighted(display, 1.0, color_mask, 0.4, 0)
                                cv2.polylines(display, [new_contour], True, (255, 0, 0), 2)
                                
                                # Cập nhật tọa độ 3D và góc gắp
                                pose = self._extract_3d_pose(new_contour)
                                if pose:
                                    cx, cy, cz, qx, qy, qz, qw = pose
                                    name = f"{obj_id}"
                                    
                                    # Lọc làm mượt tín hiệu EMA (Exponential Moving Average)
                                    ALPHA = 0.25 # 25% mới, 75% cũ
                                    if "smooth_pose" not in obj:
                                        obj["smooth_pose"] = np.array([cx, cy, cz, qx, qy, qz, qw])
                                    else:
                                        old_pose = obj["smooth_pose"]
                                        # Tịnh tiến
                                        obj["smooth_pose"][:3] = ALPHA * np.array([cx, cy, cz]) + (1 - ALPHA) * old_pose[:3]
                                        # Xoay (đảo dấu quaternion nếu cần để đi đường ngắn nhất)
                                        q_new = np.array([qx, qy, qz, qw])
                                        q_old = old_pose[3:]
                                        if np.dot(q_new, q_old) < 0: q_new = -q_new
                                        q_blend = ALPHA * q_new + (1 - ALPHA) * q_old
                                        q_blend /= np.linalg.norm(q_blend) # Chuẩn hóa
                                        obj["smooth_pose"][3:] = q_blend
                                        
                                    scx, scy, scz, sqx, sqy, sqz, sqw = obj["smooth_pose"]
                                    
                                    # Broadcast liên tục lên cây TF2 !
                                    self.node.broadcast_tf(name, scx, scy, scz, sqx, sqy, sqz, sqw)
                                    
                                    u, v = obj["center_2d"]
                                    cv2.circle(display, (u, v), 5, (0, 0, 255), -1)
                                    cv2.putText(display, f"TF: obj_{name} (Z:{cz:.3f})", (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                    cv2.putText(display, "[CUTIE TRACKING MODE] Broadcasting to TF2... (Press 'c' to Stop)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # =======================================================
                # 🖱️ CHẾ ĐỘ CHỜ NGƯỜI DÙNG CLICK CHỌN
                # =======================================================
                else:
                    for i, obj in enumerate(self.confirmed_objects):
                        cv2.polylines(display, [obj["contour"]], True, (255, 0, 0), 2) 
                        u, v = obj["center_2d"]
                        cv2.putText(display, f"#{i+1} SAVED", (u+15, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if self.temp_selected:
                        cv2.polylines(display, [self.temp_selected["contour"]], True, (0, 255, 0), 3) 
                        
                    cv2.putText(display, "[SELECT MODE] Click -> (+) Add -> (Enter) Track & TF2 -> (c) Clear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow(WIN_NAME, display)
                key = cv2.waitKey(20) & 0xFF
                
                # Các phím chức năng điều khiển
                if key == ord('+') and self.temp_selected is not None and not self.is_tracking:
                    self.confirmed_objects.append(self.temp_selected)
                    self.temp_selected = None
                    self.node.get_logger().info(f"Đã lưu vật thể. (Tổng: {len(self.confirmed_objects)})")
                    
                elif key in [13, 10]: # Bấm ENTER
                    if len(self.confirmed_objects) > 0:
                        self.is_tracking = True
                        self.temp_selected = None
                        self.node.get_logger().info("ĐÃ KHÓA DANH SÁCH! Bắt đầu TRACKING BẰNG CUTIE và BROADCAST liên tục lên TF2!")
                        
                        # Nạp tất cả Mask vào Cutie
                        self.processor = InferenceCore(self.cutie_model, cfg=self.cutie_model.cfg)
                        combined_mask_np = np.zeros((COLOR_H, COLOR_W), dtype=np.uint8)
                        obj_ids = []
                        for i, obj in enumerate(self.confirmed_objects):
                            obj_id = i + 1
                            obj_ids.append(obj_id)
                            cv2.drawContours(combined_mask_np, [obj["contour"]], -1, obj_id, -1)
                        
                        with torch.inference_mode():
                            mask_tensor = torch.from_numpy(combined_mask_np).to(self.device)
                            self.processor.step(frame_tensor, mask=mask_tensor, objects=obj_ids)
                        
                elif key in [ord('c'), ord('C')]:
                    self.is_tracking = False
                    self.confirmed_objects.clear()
                    self.temp_selected = None
                    self.processor.clear_memory()
                    self.node.get_logger().info("Đã xóa danh sách và dừng Tracking.")
                    
                elif key in [ord('q'), 27]: break
        finally:
            self._pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    
    # Chạy ROS2 spin ở background để liên tục xử lý các topic (TF2)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    
    app = VisionApp(node)
    app.run() # Vòng lặp chính nằm ở đây
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
