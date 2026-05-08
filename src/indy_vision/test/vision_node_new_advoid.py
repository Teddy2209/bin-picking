"""
Vision Node (ROS 2) - Intel RealSense + Dual GPU AI (DINO + SAM)
- Priority: Height + Isolation
- Collision: Tool 130x20x50, Z-gap 20mm, Grasp Shifting & Rz Rotation
"""
import os, sys, json, threading, math
import numpy as np
import cv2
import open3d as o3d
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R_sci
import torch
from ultralytics import YOLO
from PIL import Image

# ============================================================
# CẤU HÌNH
# ============================================================
WIDTH    = 1280
HEIGHT   = 720
FPS      = 30
DEPTH_MIN  = 0.2
DEPTH_MAX  = 0.8
WIN_NAME   = "Vision Node"
INTRINSIC_FILE = "/home/apicoo-ai/pmg/test_ros2/data_calib_intel/camera_intrinsics.json"

# ============================================================
# PRIORITY ENGINE
# ============================================================
class Priority:
    def __init__(self):
        self.w_h = 0.3
        self.w_i = 0.7

    def sort(self, objects):
        if not objects: return []
        for i, obj in enumerate(objects):
            h_score = 1.0 / (obj["pos"][2] + 0.01)
            min_dist = 1.0
            for j, other in enumerate(objects):
                if i == j: continue
                dist = np.linalg.norm(np.array(obj["pos"]) - np.array(other["pos"]))
                if dist < min_dist: min_dist = dist
            obj["priority_score"] = h_score * self.w_h + min_dist * self.w_i
        objects.sort(key=lambda x: x["priority_score"], reverse=True)
        return objects

# ============================================================
# COLLISION AVOIDANCE (Nâng cao: Shifting & Rotation)
# ============================================================
class CollisionAvoidance:
    def __init__(self):
        self.TOOL_W = 0.130 # 130mm
        self.TOOL_L = 0.020 # 20mm
        self.TOOL_H = 0.050 # 50mm

    def get_tool_rect(self, pos, quat, w, l):
        r = R_sci.from_quat(quat).as_matrix()
        tx = r[:2, 0] 
        ty = r[:2, 1]
        p = np.array(pos[:2])
        pts = [p + tx*w/2 + ty*l/2, p + tx*w/2 - ty*l/2, p - tx*w/2 - ty*l/2, p - tx*w/2 + ty*l/2]
        return np.array(pts, dtype=np.float32)

    def check_sat_collision(self, rect1, rect2):
        for poly in [rect1, rect2]:
            for i in range(len(poly)):
                p1, p2 = poly[i], poly[(i + 1) % len(poly)]
                normal = np.array([-(p2[1]-p1[1]), p2[0]-p1[0]])
                if np.linalg.norm(normal) < 1e-6: continue
                min1, max1 = np.min(rect1 @ normal), np.max(rect1 @ normal)
                min2, max2 = np.min(rect2 @ normal), np.max(rect2 @ normal)
                if max1 < min2 or max2 < min1: return False
        return True

    def is_safe(self, test_pos, test_quat, target_obj, all_objs):
        # Tool rect trong he toa do XY metric (met) - camera frame
        rect_tool = self.get_tool_rect(test_pos, test_quat, self.TOOL_W, self.TOOL_L)
        for obs in all_objs:
            if obs is target_obj: continue
            
            # 🚀 BỎ QUA BOX: Không tính va chạm với thùng chứa để có thể gắp vật bên trong
            if obs["name"].lower() == "box":
                continue
                
            # Z logic (camera frame: Z nho = gan camera hon = cao hon so voi mat ban)
            # obs["z_max"] = min(Z) = dinh cao nhat cua vat can
            if test_pos[2] < obs["z_max"] - 0.005:
                continue  # Vat can nay thap hon TCP, khong the va cham
            
            obs_box = obs.get("metric_box")
            if obs_box is not None and self.check_sat_collision(rect_tool, obs_box):
                return False
        return True

    def find_safe_grasp(self, target_obj, all_objs):
        # 1. Thu vi tri mac dinh (tam vat)
        if self.is_safe(target_obj["pos"], target_obj["quat"], target_obj, all_objs):
            return target_obj["pos"], target_obj["quat"]

        # 2. Tim diem ho neu bi va cham
        if target_obj.get("is_symmetric", False):
            # Hinh tron / vuong: Xoay tool theo Rz de tim goc ho
            orig_r = R_sci.from_quat(target_obj["quat"])
            for angle in range(10, 180, 10):
                for d in [1, -1]:
                    new_q = (R_sci.from_euler('z', angle * d, degrees=True) * orig_r).as_quat()
                    if self.is_safe(target_obj["pos"], new_q, target_obj, all_objs):
                        return target_obj["pos"], new_q.tolist()
        else:
            # Vat dai: Truot TCP doc theo truc Y cua tool (mui ten xanh la)
            # grasp_Y = cot thu 2 cua ma tran xoay = chinh xac la huong mui ten Y hien thi
            r_mat = R_sci.from_quat(target_obj["quat"]).as_matrix()
            shift_dir = r_mat[:, 1]  # grasp_Y = huong truot (doc chieu dai vat)

            # Gioi han truot = 40% chieu dai vat de khong truot ra ngoai vat
            max_shift = target_obj["dims"][0] * 0.4

            # Thu tung buoc truot: nho -> lon, ca 2 chieu (+/-) theo mui ten Y
            shift_steps = []
            step = 0.01  # 1cm moi buoc
            s = step
            while s <= max_shift + 1e-6:
                shift_steps.extend([s, -s])
                s += step

            for shift in shift_steps:
                new_pos = np.array(target_obj["pos"]) + shift_dir * shift
                if self.is_safe(new_pos, target_obj["quat"], target_obj, all_objs):
                    return new_pos.tolist(), target_obj["quat"]

        return None, None

# ============================================================
# POSE STABILIZER (Chong nhay toa do)
# ============================================================
class PoseStabilizer:
    def __init__(self, alpha=0.3, deadband_m=0.005, stable_frames=5):
        # alpha: he so EMA (thap = mo hon, chong nhay nhung tri hoan)
        # deadband_m: chi cap nhat neu thay doi > 5mm
        # stable_frames: chi publish sau N frame on dinh lien tiep
        self.alpha = alpha
        self.deadband = deadband_m
        self.stable_frames = stable_frames
        self._pos = None       # Vi tri da lam muot
        self._quat = None
        self._name = None
        self._raw_pos = None   # Vi tri raw moi nhat
        self._count = 0        # So frame on dinh lien tiep
        self._confirmed_pos = None   # Vi tri da confirm gui robot
        self._confirmed_quat = None
        self._confirmed_name = None

    def update(self, name, pos, quat):
        pos = np.array(pos)
        # Neu doi ten vat thi reset
        if name != self._name:
            self._name = name
            self._pos = pos.copy()
            self._quat = np.array(quat)
            self._raw_pos = pos.copy()
            self._count = 0
            return False  # Chua on dinh

        # EMA lam muot vi tri
        self._pos = self.alpha * pos + (1 - self.alpha) * self._pos

        # Kiem tra deadband so voi confirmed pos
        ref = self._confirmed_pos if self._confirmed_pos is not None else self._pos
        change = np.linalg.norm(pos - ref)

        if change < self.deadband:
            # Vi tri on dinh, tang dem
            self._count += 1
        else:
            # Thay doi lon, reset dem
            self._count = 0
            self._quat = np.array(quat)

        # Confirm va cho phep publish khi on dinh du frame
        if self._count >= self.stable_frames:
            self._confirmed_pos = self._pos.copy()
            self._confirmed_quat = self._quat.copy()
            self._confirmed_name = name
            return True  # San sang gui robot

        return False  # Chua on dinh

    def get_confirmed(self):
        if self._confirmed_pos is not None:
            return (
                self._confirmed_name,
                self._confirmed_pos.tolist(),
                self._confirmed_quat.tolist()
            )
        return None, None, None

    def get_smoothed(self):
        if self._pos is not None:
            return self._pos.tolist(), (self._quat.tolist() if self._quat is not None else None)
        return None, None

    def reset(self):
        self._name = None
        self._pos = None
        self._count = 0
        self._confirmed_pos = None

# ============================================================
# ROS 2 NODE
# ============================================================
class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.task_pub = self.create_publisher(String, '/vision/pick_task', 10)

    def broadcast_tf(self, name, pos, quat):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "link_camera"
        t.child_frame_id = f"target_{name}"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)
    
# ============================================================
# AI ENGINE (Dual GPU)
# ============================================================
class AIEngine:
    def __init__(self, logger):
        self.logger = logger
        # Đường dẫn tới model YOLOv11-seg vừa train
        self.MODEL_PATH = "/home/apicoo-ai/pmg/test_ros2/runs/segment/runs/easy_bin_picking_experiment/weights/best.pt"
        self.model = None
        threading.Thread(target=self._load_model, daemon=True).start()
 
    def _load_model(self):
        try:
            # Load model YOLOv11-seg lên GPU 0
            from ultralytics import YOLO
            self.model = YOLO(self.MODEL_PATH)
            self.model.to('cuda:0')
            self.logger.info(f"✅ YOLOv11 Engine Ready. Classes: {self.model.names}")
        except Exception as e: 
            self.logger.error(f"❌ AI Load Error: {e}")
 
    def get_mask_contour(self, img_bgr):
        if self.model is None: return None
        try:
            # 1. Chạy Inference với YOLOv11
            results = self.model.predict(img_bgr, conf=0.6, device='cuda:0', verbose=False)
            
            detected = []
            if results[0].masks is not None:
                # Lấy masks raw từ GPU
                raw_masks = results[0].masks.data.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                # Kích thước ảnh gốc
                orig_h, orig_w = img_bgr.shape[:2]
                
                for i, m in enumerate(raw_masks):
                    cls_id = int(classes[i])
                    class_name = self.model.names[cls_id]
                    
                    # 🚀 QUAN TRỌNG: Resize mask về đúng kích thước ảnh gốc (1280x720)
                    # YOLO thường trả về mask có kích thước chia hết cho 32 (vd 736)
                    m_resized = cv2.resize(m, (orig_w, orig_h))
                    
                    detected.append({
                        "name": class_name, 
                        "mask": (m_resized > 0.5)
                    })
            return detected
        except Exception as e:
            self.logger.error(f"AI Predict Error: {e}")
            return None

# ============================================================
# VISION UI & APP
# ============================================================
class VisionUI:
    def __init__(self, node: VisionNode):

        # Khởi tạo các đối tượng
        try:
            self.logger = node.get_logger()
            self.node = node
            self.ai = AIEngine(self.logger)
            self.priority = Priority()
            self.collision = CollisionAvoidance()
            self.stabilizer = PoseStabilizer(alpha=0.3, deadband_m=0.005, stable_frames=5)
        except Exception as e:
            self.node.get_logger().error(f"Lỗi load AI: {e}")

        # Khởi tạo các biến
        #self.trackers = []  
        #self.temp_selected = None 
        #self.confirmed_objects = [] 
        #self.is_tracking = False 
        self._custom_intrinsics = self._load_custom_intrinsics()

        # Khởi động RealSense ngay lập tức
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
        profile = self._pipeline.start(cfg)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1) # 0 Tắt auto-exposure, 1 Bật auto-exposure

        self._rs_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._spatial = rs.spatial_filter()
        self._hole    = rs.hole_filling_filter()
        self._colmap  = rs.colorizer()
        self._align   = rs.align(rs.stream.color)

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 720)
        self._color_img = None
        self._depth_raw = None
        self._pc = rs.pointcloud() # Công cụ tạo PointCloud của RealSense

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
                self.node.get_logger().info("Đã load custom intrinsics.")
            except Exception as e:
                self.node.get_logger().warn(f"Failed custom intrinsics: {e}")

    def _extract_3d_pose(self, mask_bool, verts_3d, plane_normal):
        """
        Tính toán chi tiết: Dùng rs2_deproject_pixel_to_point để trích xuất 3D từ Mask
        """
        # 1. Trích xuất tọa độ pixel từ Mask
        ys, xs = np.where(mask_bool > 0)
        if len(xs) < 10: return None
        
        # 2. Lấy mẫu (Subsampling) để đảm bảo tốc độ (Lấy khoảng 1500-2000 điểm là đủ cho PCA)
        step = max(1, len(xs) // 2000)
        xs_s, ys_s = xs[::step], ys[::step]
        
        intr = self._custom_intrinsics if self._custom_intrinsics is not None else self._rs_intrinsics
        pts_list = []
        for u, v in zip(xs_s, ys_s):
            d = self._depth_raw.get_distance(int(u), int(v))
            if DEPTH_MIN < d < DEPTH_MAX:
                # Chiếu ngược từ pixel 2D sang point 3D bằng Intrinsics chuẩn
                pt = rs.rs2_deproject_pixel_to_point(intr, [int(u), int(v)], d)
                pts_list.append(pt)
        
        if len(pts_list) < 10: return None
        valid_pts = np.array(pts_list)
        
        # Lọc nhiễu Outlier (Z-median)
        z_values = valid_pts[:, 2]
        z_median = np.median(z_values)
        valid_pts = valid_pts[np.abs(z_values - z_median) < 0.03]
        if len(valid_pts) < 5: return None
            
        # 2. Tâm hình học & Điểm cao nhất
        cx, cy, cz = np.mean(valid_pts, axis=0)
        z_max = float(np.min(valid_pts[:, 2]))
        
        # 3. Tính toán ma trận xoay (Align) để mặt phẳng bàn song song với mặt phẳng XY
        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(plane_normal, z_axis)
        c = np.dot(plane_normal, z_axis)
        s = np.linalg.norm(v)
        
        if s > 1e-6:
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_align = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        else:
            R_align = np.eye(3)
            
        # 🚀 Lấy TOÀN BỘ điểm của vật thể để tính toán Box khít nhất
        full_pts = verts_3d[mask_bool > 0]
        full_valid = full_pts[(full_pts[:, 2] > DEPTH_MIN) & (full_pts[:, 2] < DEPTH_MAX)]
        
        if len(full_valid) > 0:
            full_aligned = full_valid @ R_align.T
            min_p = np.min(full_aligned, axis=0)
            max_p = np.max(full_aligned, axis=0)
            
            # Tâm hình học chuẩn từ Full PCL
            center_aligned = (min_p + max_p) / 2
            cx, cy, cz = center_aligned @ R_align
            
            # Kích thước thực (L, W, H) chuẩn từ Full PCL
            L = float(max_p[0] - min_p[0])
            W = float(max_p[1] - min_p[1])
            H = float(max_p[2] - min_p[2])
        else:
            cx, cy, cz, L, W, H = 0, 0, 0, 0, 0, 0

        # 4. Thực hiện PCA 2D trên các điểm mẫu để tìm hướng (Giữ tốc độ nhanh)
        pts_aligned = valid_pts @ R_align.T
        pts_xy = pts_aligned[:, :2]
        pts_centered = pts_xy - np.mean(pts_xy, axis=0)
        cov_xy = np.cov(pts_centered, rowvar=False)
        val_xy, vec_xy = np.linalg.eigh(cov_xy)
        
        idx = val_xy.argsort()[::-1]
        vec_xy = vec_xy[:, idx] # vec_xy[:, 0] là hướng chính (Trục X của vật thể)

        # 🚀 TÌM KÍCH THƯỚC KHÍT NHẤT (Oriented Bounding Box - OBB)
        # Xoay toàn bộ Full PCL theo hướng PCA để tìm L, W chuẩn nhất
        R_pca = np.eye(3)
        R_pca[0, 0] = vec_xy[0, 0]; R_pca[0, 1] = vec_xy[1, 0]
        R_pca[1, 0] = -vec_xy[1, 0]; R_pca[1, 1] = vec_xy[0, 0]
        
        full_oriented = full_aligned @ R_pca.T
        min_o = np.min(full_oriented, axis=0)
        max_o = np.max(full_oriented, axis=0)
        
        L = float(max_o[0] - min_o[0])
        W = float(max_o[1] - min_o[1])
        H = float(max_o[2] - min_o[2])
        
        # Tâm hình học chuẩn của OBB
        center_oriented = (min_o + max_o) / 2
        real_center = (center_oriented @ R_pca) @ R_align
        cx, cy, cz = real_center


        # 5. Phân tích hình học từ Mask 2D (Contour)
        cnts, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        contour = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        is_circular = False
        if perimeter > 0:
            if (4 * np.pi * area / (perimeter * perimeter)) > 0.75:
                is_circular = True
        
        aspect_ratio = L / W if W > 0.001 else 10.0
        is_symmetric = is_circular or (aspect_ratio < 1.3)

        # 6. Xây dựng Ma trận xoay 3D (Grasp Pose)
        # Robot tiếp cận từ trên xuống, nên trục Z của Tool = Pháp tuyến mặt bàn (Z hướng dương về phía gắp)
        grasp_Z = plane_normal
        
        if is_symmetric:
            # Chọn trục X ngẫu nhiên nhưng vuông góc với Z
            temp_X = np.array([1.0, 0.0, 0.0])
            grasp_Y = np.cross(grasp_Z, temp_X)
            if np.linalg.norm(grasp_Y) < 1e-6:
                temp_X = np.array([0.0, 1.0, 0.0])
                grasp_Y = np.cross(grasp_Z, temp_X)
            grasp_Y /= np.linalg.norm(grasp_Y)
            grasp_X = np.cross(grasp_Y, grasp_Z)
        else:
            # Hướng X trong mặt phẳng chuẩn (sau PCA)
            v_major_aligned = np.array([vec_xy[0, 0], vec_xy[1, 0], 0.0])
            # Xoay ngược v_major về không gian 3D thực tế
            grasp_X = v_major_aligned @ R_align
            
            # Ép hướng X luôn nghiêng về bên phải camera để Robot không bị xoay cổ ngẫu nhiên
            if grasp_X[0] < 0: grasp_X = -grasp_X
            
            # Tính Y và hiệu chỉnh lại X để đảm bảo trực giao tuyệt đối
            grasp_Y = np.cross(grasp_Z, grasp_X)
            grasp_Y /= np.linalg.norm(grasp_Y)
            grasp_X = np.cross(grasp_Y, grasp_Z)

        rot_mat = np.column_stack((grasp_X, grasp_Y, grasp_Z))
        quat = R_sci.from_matrix(rot_mat).as_quat()

        # Metric box (Cho xử lý va chạm trên mặt bàn)
        pts_xy_m = pts_aligned[:, :2].astype(np.float32)
        metric_box = cv2.boxPoints(cv2.minAreaRect(pts_xy_m)).astype(np.float32)

        return {
            "pos": [float(cx), float(cy), float(cz)],
            "quat": [float(q) for q in quat],
            "z_max": z_max,
            "dims": [L, W, H],
            "is_circular": is_circular,
            "is_symmetric": is_symmetric,
            "metric_box": metric_box,
            "oriented_box": box_pts,
            "contour": contour
        }
    def _project_3d_to_2d(self, pt_3d):
        """Chiếu một điểm 3D xuống pixel 2D dùng camera intrinsics."""
        intr = self._rs_intrinsics
        x, y, z = pt_3d
        if z < 1e-6: return None
        u = int(intr.fx * x / z + intr.ppx)
        v = int(intr.fy * y / z + intr.ppy)
        return (u, v)

    def _draw_grasp_viz(self, display, obj):
        """Vẽ tâm gắp (TCP) và box tool 130x20mm mô phỏng trên ảnh."""
        pos = np.array(obj["safe_pos"])
        quat = obj["safe_quat"]
        r = R_sci.from_quat(quat).as_matrix()
        tx = r[:, 0]  # Trục X của tool (chiều dài 130mm)
        ty = r[:, 1]  # Trục Y của tool (chiều rộng 20mm)

        # 1. Tính 4 đỉnh của box tool trong 3D
        W, L = 0.130 / 2, 0.020 / 2  # Half-sizes
        corners_3d = [
            pos + tx * W + ty * L,
            pos + tx * W - ty * L,
            pos - tx * W - ty * L,
            pos - tx * W + ty * L,
        ]

        # 2. Chiếu xuống 2D
        corners_2d = [self._project_3d_to_2d(c) for c in corners_3d]
        tcp_2d = self._project_3d_to_2d(pos)

        # 3. Vẽ box tool (màu vàng, độ dày 2)
        if all(c is not None for c in corners_2d):
            pts = np.array(corners_2d, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)  # Vàng

        # 4. Vẽ tâm gắp TCP (dấu chữ thập lớn)
        if tcp_2d:
            cx, cy = tcp_2d
            cv2.drawMarker(display, (cx, cy), (0, 255, 255),
                           cv2.MARKER_CROSS, 30, 3)
            cv2.circle(display, (cx, cy), 6, (0, 255, 255), -1)

            # 5. Vẽ trục X (màu đỏ - chiều dài khớp)
            x_tip = self._project_3d_to_2d(pos + tx * 0.06)
            if x_tip:
                cv2.arrowedLine(display, (cx, cy), x_tip, (0, 0, 255), 2, tipLength=0.3)
                cv2.putText(display, "X", x_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 6. Vẽ trục Y (màu xanh lá - chiều rộng khớp)
            y_tip = self._project_3d_to_2d(pos + ty * 0.04)
            if y_tip:
                cv2.arrowedLine(display, (cx, cy), y_tip, (0, 255, 0), 2, tipLength=0.3)
                cv2.putText(display, "Y", y_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def visualize_3d(self, color_img, depth_frame, objects):
        """
        Dựng cửa sổ 3D hiển thị PointCloud và các Bounding Box 3D của vật thể.
        """
        try:
            print("📊 Đang khởi tạo cửa sổ 3D...")
            # 1. Tạo PointCloud toàn cảnh
            points = self._pc.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts)
            
            # Tô màu cho PointCloud
            color_np = color_img.astype(np.float32) / 255.0
            color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
            color_flat = color_np.reshape(-1, 3)

            # 🚀 ĐỒNG BỘ MÀU MASK LÊN PCL
            for obj in objects:
                if obj["name"].lower() == "box":
                    if obj.get("bottom_mask") is not None and obj.get("wall_mask") is not None:
                        # Đáy hộp màu đen xám, Thành hộp màu đỏ
                        color_flat[obj["bottom_mask"].flatten()] = [0.15, 0.15, 0.15]
                        color_flat[obj["wall_mask"].flatten()] = [1.0, 0.0, 0.0]
                    else:
                        color_flat[obj["mask"].flatten()] = [1.0, 0.0, 0.0] # Fallback đỏ
                    
                    # Vẽ thêm một khung bao (Box) ảo cho thành hộp
                    continue
                    
                # Tạo màu ngẫu nhiên dựa trên tên cho các vật thể khác
                state = np.random.get_state()
                np.random.seed(abs(hash(obj["name"])) % 1000)
                color_rgb = np.random.randint(0, 255, 3).tolist()
                np.random.set_state(state)
                
                # Chuyển BGR sang RGB và chuẩn hóa về [0, 1] cho Open3D
                color_norm = [color_rgb[2]/255.0, color_rgb[1]/255.0, color_rgb[0]/255.0]
                
                # Lấy mặt nạ phẳng
                m_flat = obj["mask"].flatten()
                
                # Tô màu cho những điểm thuộc vật thể này (Pha trộn 20% để vẫn thấy vân bề mặt)
                color_flat[m_flat] = color_flat[m_flat] * 0.2 + np.array(color_norm) * 0.8

            pcd.colors = o3d.utility.Vector3dVector(color_flat)
            
            # Lọc bớt điểm ở xa để nhìn rõ hơn (vd > 1.5m)
            pcd = pcd.select_by_index(np.where(verts[:, 2] < 1.5)[0])
            pcd = pcd.voxel_down_sample(0.002) # Downsample để xoay mượt hơn

            geometries = [pcd]

            # 2. Thêm các Bounding Box 3D của vật thể
            for obj in objects:
                # Tạo OrientedBoundingBox từ dữ liệu PCA
                center = np.array(obj["pos"])
                R = R_sci.from_quat(obj["quat"]).as_matrix()
                extent = np.array(obj["dims"])
                
                obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
                
                # Màu sắc ngẫu nhiên cho mỗi vật thể
                state = np.random.get_state()
                np.random.seed(abs(hash(obj["name"])) % 1000)
                color = np.random.rand(3)
                np.random.set_state(state)
                
                obb.color = color
                geometries.append(obb)
                
                # Thêm hệ trục tọa độ tại tâm vật thể
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=center)
                mesh_frame.rotate(R, center=center)
                geometries.append(mesh_frame)

            # 3. Hiển thị
            o3d.visualization.draw_geometries(geometries, 
                                            window_name="3D Vision Inspection",
                                            width=1280, height=720,
                                            mesh_show_back_face=True)
        except Exception as e:
            print(f"❌ Lỗi 3D Viz: {e}")

    def run(self):
        ai_result = None
        try:
            while rclpy.ok():
                try:
                    frames = self._pipeline.wait_for_frames()
                except Exception:
                    frames = None

                if frames and frames.size() >= 2:
                    aligned = self._align.process(frames)
                    df, cf = aligned.get_depth_frame(), aligned.get_color_frame()
                    if df and cf:
                        df = self._spatial.process(df)
                        df = self._hole.process(df)
                        self._depth_raw = df.as_depth_frame()
                        self._color_img = np.asanyarray(cf.get_data())
                        
                        # 🚀 TẠO POINTCLOUD TOÀN CẢNH
                        points = self._pc.calculate(self._depth_raw)
                        verts_3d = np.asanyarray(points.get_vertices()).view(np.float32).reshape((HEIGHT, WIDTH, 3))
                        
                        # BƯỚC 1: AI NHẬN DIỆN
                        ai_result = self.ai.get_mask_contour(self._color_img)
                        
                        if ai_result:
                            # 🚀 MỞ RỘNG TOÀN BỘ MASK THÊM 11 PIXEL (Bao gồm cả Box và Linh kiện)
                            kernel_11 = np.ones((11, 11), np.uint8)
                            for obj in ai_result:
                                obj["mask"] = cv2.dilate(obj["mask"].astype(np.uint8), kernel_11, iterations=1).astype(bool)
                        
                        # 🚀 TÌM MẶT PHẲNG ĐÁY HỘP TỪ MASK CỦA BOX (RANSAC)
                        plane_normal = np.array([0.0, 0.0, 1.0]) # Mặc định
                        if ai_result:
                            box_obj = next((o for o in ai_result if o["name"].lower() == "box"), None)
                            if box_obj:
                                box_mask = box_obj["mask"].copy()
                                
                                # 🚀 BƯỚC QUAN TRỌNG: Lấy Mask Box trừ đi các Mask vật thể
                                # Mask vật thể đã được mở rộng 11 pixel ở trên, giờ mở rộng thêm 19 pixel nữa (tổng ~30)
                                kernel_sub = np.ones((19, 19), np.uint8)
                                for o in ai_result:
                                    if o["name"].lower() != "box":
                                        # Dùng mask nong to để khoét lỗ lớn hơn trên mặt hộp
                                        obj_mask_expanded = cv2.dilate(o["mask"].astype(np.uint8), kernel_sub, iterations=1)
                                        box_mask[obj_mask_expanded > 0] = False
                                
                                ys, xs = np.where(box_mask > 0)
                                box_pts = verts_3d[ys, xs]
                                
                                # Lọc điểm hợp lệ
                                valid_idx = np.where((box_pts[:, 2] > 0.1) & (box_pts[:, 2] < 2.0))[0]
                                valid_verts = box_pts[valid_idx]
                                
                                if len(valid_verts) > 1000:
                                    pcd_tmp = o3d.geometry.PointCloud()
                                    pcd_tmp.points = o3d.utility.Vector3dVector(valid_verts)
                                    pcd_tmp = pcd_tmp.voxel_down_sample(0.015) # Downsample
                                    try:
                                        plane_model, _ = pcd_tmp.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=200)
                                        a, b, c, d = plane_model
                                        if c < 0: a, b, c, d = -a, -b, -c, -d # Luôn hướng lên/về phía camera
                                        plane_normal = np.array([a, b, c])
                                        plane_normal /= np.linalg.norm(plane_normal)
                                        
                                        # Phân loại Đáy hộp và Thành hộp bằng khoảng cách tới mặt phẳng
                                        dists = np.abs(np.dot(valid_verts, plane_normal) + d)
                                        is_bottom = dists < 0.05 # Dưới 5cm là đáy
                                        
                                        bottom_mask = np.zeros_like(box_mask, dtype=bool)
                                        wall_mask = np.zeros_like(box_mask, dtype=bool)
                                        
                                        # Gán True cho các pixel thuộc đáy và thành
                                        bottom_mask[ys[valid_idx[is_bottom]], xs[valid_idx[is_bottom]]] = True
                                        wall_mask[ys[valid_idx[~is_bottom]], xs[valid_idx[~is_bottom]]] = True
                                        
                                        box_obj["bottom_mask"] = bottom_mask
                                        box_obj["wall_mask"] = wall_mask
                                    except Exception as e:
                                        pass
                else:
                    ai_result = None

                if self._color_img is None:
                    continue

                display = self._color_img.copy()

                # BƯỚC 2: TRÍCH XUẤT POSE 3D CHO TẤT CẢ
                processed_objs = []
                if ai_result:
                    for obj in ai_result:
                        mask_bool = obj["mask"]
                        p = self._extract_3d_pose(mask_bool, verts_3d, plane_normal)
                        if p:
                            processed_objs.append({
                                "name": obj["name"],
                                "mask": mask_bool,
                                "contour": p["contour"],
                                "pos": p["pos"],
                                "quat": p["quat"],
                                "v_major": None,
                                "z_max": p["z_max"],
                                "dims": p["dims"],
                                "is_circular": p["is_circular"],
                                "is_symmetric": p["is_symmetric"],
                                "metric_box": p["metric_box"],
                                "oriented_box": p["oriented_box"],
                                "bottom_mask": obj.get("bottom_mask"),
                                "wall_mask": obj.get("wall_mask"),
                            })

                # BƯỚC 3: SẮP XẾP ƯU TIÊN
                sorted_objs = self.priority.sort(processed_objs)

                # BƯỚC 4: KIỂM TRA VA CHẠM & CHỌN MỤC TIÊU
                best_target = None
                for obj in sorted_objs:
                    # 🚀 LOẠI BOX: Không bao giờ gắp thùng chứa
                    if obj["name"].lower() == "box":
                        continue
                        
                    safe_pos, safe_quat = self.collision.find_safe_grasp(obj, sorted_objs)
                    if safe_pos:
                        obj["safe_pos"] = safe_pos
                        obj["safe_quat"] = safe_quat
                        best_target = obj
                        break

                # BƯỚC 5: LÀM MỊN + PUBLISH KHI ỔN ĐỊNH
                if best_target:
                    # Làm mượt tọa độ qua EMA
                    is_stable = self.stabilizer.update(
                        best_target["name"],
                        best_target["safe_pos"],
                        best_target["safe_quat"]
                    )
                    # --- DIAGNOSTIC LOG ---
                    print(f"[STAB] name={best_target['name']} count={self.stabilizer._count}/{self.stabilizer.stable_frames} stable={is_stable}")
                    # ----------------------
                    smooth_pos, smooth_quat = self.stabilizer.get_smoothed()

                    # TF2: luôn broadcast vị trí đã làm mượt (để Rviz hiển thị ổn định)
                    if smooth_pos and smooth_quat:
                        self.node.broadcast_tf(best_target["name"], smooth_pos, smooth_quat)
                        best_target["safe_pos"] = smooth_pos  # Dùng cho viz

                    # Topic robot: chỉ gửi khi đã ổn định đủ frame
                    if is_stable:
                        c_name, c_pos, c_quat = self.stabilizer.get_confirmed()
                        self.node.task_pub.publish(String(data=json.dumps({
                            "name": c_name,
                            "pos": c_pos,
                            "quat": c_quat,
                            "dims": best_target["dims"],
                        })))
                        print(f"[STAB] >>> PUBLISHED task for {c_name}")
                else:
                    self.stabilizer.reset()

                # BƯỚC 6: HIỂN THỊ
                for rank, obj in enumerate(sorted_objs, 1):
                    label_name = obj["name"]
                    is_target = (obj is best_target)

                    state = np.random.get_state()
                    np.random.seed(abs(hash(label_name)) % 1000)
                    color = np.random.randint(0, 255, 3).tolist()
                    np.random.set_state(state)

                    display[obj["mask"]] = display[obj["mask"]] * 0.5 + np.array(color) * 0.5
                    cv2.drawContours(display, [obj["contour"]], -1, color, 3)

                    box_color = (0, 255, 0) if is_target else (255, 255, 255)
                    cv2.polylines(display, [obj["oriented_box"]], True, box_color, 3 if is_target else 1)

                    status = "[TARGET]" if is_target else ""
                    label = f"#{rank} {status} {label_name} Z:{obj['pos'][2]:.3f}"
                    u = int(np.mean(obj["oriented_box"][:, 0]))
                    v = int(np.mean(obj["oriented_box"][:, 1]))
                    cv2.putText(display, label, (u - 60, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    # Vẽ TCP + Box Tool mô phỏng cho vật được chọn
                    if is_target:
                        self._draw_grasp_viz(display, obj)

                if not best_target and sorted_objs:
                    cv2.putText(display, "WARNING: All objects blocked!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Hien thi trang thai on dinh goc trai tren
                cnt = self.stabilizer._count
                sf = self.stabilizer.stable_frames
                conf = self.stabilizer._confirmed_pos
                if best_target:
                    if conf is not None and cnt >= sf:
                        stab_txt = f"[LOCKED] Sending to Robot"
                        stab_color = (0, 255, 0)
                    else:
                        stab_txt = f"[STABILIZING] {min(cnt, sf)}/{sf}"
                        stab_color = (0, 200, 255)
                    cv2.putText(display, stab_txt, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, stab_color, 2)

                cv2.imshow(WIN_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('v'):
                    # Nhấn 'v' để xem 3D (Chạy trực tiếp trên luồng chính để đảm bảo hiện cửa sổ)
                    if self._color_img is not None and self._depth_raw is not None:
                        print("⌨️ Đã nhận phím 'v' - Đang chuẩn bị dữ liệu 3D...")
                        self.visualize_3d(self._color_img.copy(), self._depth_raw, processed_objs)
        finally:
            self._pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    
    # Chạy ROS2 spin ở background để liên tục xử lý các topic (TF2)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    app = VisionUI(node)
    app.run() # Vòng lặp chính nằm ở đây
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
