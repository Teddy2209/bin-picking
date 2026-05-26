"""
Vision Node (ROS 2) - Intel RealSense + Dual GPU AI (DINO + SAM)
- Priority: Height + Isolation
- Collision: Tool 130x20x50, Z-gap 20mm, Grasp Shifting & Rz Rotation
"""
import os, sys, json, threading, math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R_sci
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from dataclasses import dataclass

# ============================================================
# CẤU HÌNH
# ============================================================
WIDTH    = 1280
HEIGHT   = 720
FPS      = 30
DEPTH_MIN  = 0.1
DEPTH_MAX  = 2.0
WIN_NAME   = "Vision Node"
INTRINSIC_FILE = "/home/apicoo-ai/pmg/bin_picking_ws/src/indy_vision/data_calib_intel/camera_intrinsics.json"

@dataclass
class DetectedObject:
    name: str
    pos: list
    quat: list
    z_max: float
    dims: list

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
            # Z logic (camera frame: Z nho = gan camera hon = cao hon so voi mat ban)
            # obs["z_max"] = min(Z) = dinh cao nhat cua vat can
            # Neu TCP (test_pos[2]) < obs_z_max - 0.005: 
            #   -> TCP dang o cao hon dinh vat can it nhat 5mm -> An toan
            if test_pos[2] < obs["z_max"] - 0.005:
                continue  # Vat can nay thap hon TCP, khong the va cham
            # Neu TCP bang hoac lon hon (sau hon) -> co nguy co va cham -> check SAT
            # Neu khong, co kha nang va cham theo Z -> kiem tra SAT 2D
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
        self.sub = self.create_subscription(String, '/robot/state', self.robot_state_callback, 10)
        self.lock = False

    def broadcast_tf(self, name, pos, quat):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "link_camera"
        t.child_frame_id = f"target_{name}"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)
    def robot_state_callback(self, msg):
        self.get_logger().info(f"Robot state: {msg.data}")
        if msg.data == "moving":
            self.lock = True
        else:
            self.lock = False

# ============================================================
# AI ENGINE (Dual GPU)
# ============================================================
class AIEngine:
    def __init__(self, logger):
        self.logger = logger
        self.MODEL_SAM_PATH = "sam2_l.pt"
        self.MODEL_DINO_PATH = "IDEA-Research/grounding-dino-base"
        self.processor_DINO = None
        self.DINO_model = None
        self.SAM_model = None
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            from ultralytics import SAM
            with torch.cuda.device(0):
                self.processor_DINO = AutoProcessor.from_pretrained(self.MODEL_DINO_PATH)
                self.DINO_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_DINO_PATH).to('cuda:0')
            with torch.cuda.device(1):
                self.SAM_model = SAM(self.MODEL_SAM_PATH)
                if hasattr(self.SAM_model, 'model'): self.SAM_model.model.to('cuda:1')
            self.logger.info("AI Multi-GPU Ready.")
        except Exception as e: self.logger.error(f"AI Load Error: {e}")

    def get_mask_contour(self, img_bgr):
        if not self.SAM_model: return None
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            queries = "a rubik's cube . pliers . blue block . box . beer can . bolt"
            with torch.cuda.device(0):
                inputs = self.processor_DINO(images=pil_img, text=queries, return_tensors="pt").to('cuda:0')
                with torch.no_grad(): outputs = self.DINO_model(**inputs)
            results = self.processor_DINO.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0.40,text_threshold=0.20, target_sizes=[img_bgr.shape[:2]])[0]
            detected = []
            if len(results["boxes"]) > 0:
                with torch.cuda.device(1):
                    res_sam = self.SAM_model.predict(img_bgr, bboxes=results["boxes"].cpu().numpy(), device='cuda:1', verbose=False)
                    if res_sam and res_sam[0].masks is not None:
                        masks = res_sam[0].masks.data.cpu().numpy()
                        for i, m in enumerate(masks):
                            detected.append({"name": results["labels"][i], "mask": (m > 0.5)})
            return detected
        except: return None

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
        self.trackers = []  
        self.temp_selected = None 
        self.confirmed_objects = [] 
        self.is_tracking = False 
        self._custom_intrinsics = self._load_custom_intrinsics()

        # Khởi động RealSense ngay lập tức
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)
        profile = self._pipeline.start(cfg)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 1) # Tắt auto-exposure

        self._rs_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._active_intrinsics = self._rs_intrinsics
        self._spatial = rs.spatial_filter()
        self._hole    = rs.hole_filling_filter()
        self._colmap  = rs.colorizer()
        self._align   = rs.align(rs.stream.color)

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 720)
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

    def _extract_3d_pose(self, contour):
        """
        Tính toán chi tiết: Zmax, Tâm Gắp, Hướng PCA và Oriented Box sát biên dạng
        """
        if self._depth_raw is None: return None
        
        # 1. Tạo mask 2D từ contour
        mask_img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.drawContours(mask_img, [contour], -1, 255, -1)
        ys, xs = np.where(mask_img == 255)
        
        # 2. Lấy Oriented Box 2D sát biên dạng vật (thay cho DINO Box)
        rect = cv2.minAreaRect(contour)
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        
        # 3. Lấy tập hợp điểm 3D (Subsample để nhanh)
        step = max(1, len(xs) // 1000)
        xs, ys = xs[::step], ys[::step]
        intr = getattr(self, '_custom_intrinsics', None) or self._active_intrinsics
        pts_3d = []
        for u, v in zip(xs, ys):
            d = self._depth_raw.get_distance(int(u), int(v))
            if DEPTH_MIN < d < DEPTH_MAX:
                pt = rs.rs2_deproject_pixel_to_point(intr, [int(u), int(v)], d)
                pts_3d.append(pt)
                
        if len(pts_3d) < 10: return None
        pts_array = np.array(pts_3d)
        
        # 4. Lọc nhiễu và Tìm Zmax (Điểm cao nhất - Z nhỏ nhất)
        z_values = pts_array[:, 2]
        z_max = float(np.min(z_values)) # Z nhỏ nhất là vật ở cao nhất
        z_median = np.median(z_values)
        valid_mask = np.abs(z_values - z_median) < 0.03 
        valid_pts = pts_array[valid_mask]
        if len(valid_pts) < 5: valid_pts = pts_array
            
        # Tâm hình học gắp
        cx, cy, cz = np.mean(valid_pts, axis=0)
        
        # 5. PCA tìm hướng
        cov = np.cov(valid_pts, rowvar=False)
        val, vec = np.linalg.eigh(cov)
        idx = val.argsort()[::-1]
        vec = vec[:, idx]
        
        # Circularity check
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        is_circular = False
        if perimeter > 0:
            if (4 * np.pi * area / (perimeter * perimeter)) > 0.75:
                is_circular = True

        # 7. Kich thuoc thuc (L, W, H)
        p_pca = (valid_pts - [cx, cy, cz]) @ vec
        L = float(np.max(p_pca[:, 0]) - np.min(p_pca[:, 0]))
        W = float(np.max(p_pca[:, 1]) - np.min(p_pca[:, 1]))
        H = float(np.max(valid_pts[:, 2]) - np.min(valid_pts[:, 2]))

        # Phan loai hinh dang qua aspect ratio
        aspect_ratio = L / W if W > 0.001 else 10.0
        # Hinh tron VA hinh gan vuong deu khong dung PCA huong
        is_symmetric = is_circular or (aspect_ratio < 1.5)

        # 6. Huong gap - on dinh bang snap goc 5 do
        v_major = vec[:, 0]
        if is_symmetric:
            # Hinh doi xung: dung Identity, Collision se xoay Rz tim goc ho
            grasp_X = np.array([1.0, 0.0, 0.0])
            grasp_Y = np.array([0.0, 1.0, 0.0])
            grasp_Z = np.array([0.0, 0.0, 1.0])
        else:
            # Vat dai: dung PCA nhung snap 5 do tranh nhay loan
            v_xy = vec[:2, 0]
            angle = np.arctan2(v_xy[1], v_xy[0])
            # Dua goc ve [0, pi) - giai quyet sign ambiguity PCA
            if angle < 0:
                angle += np.pi
            if angle >= np.pi:
                angle -= np.pi
            SNAP = np.deg2rad(5.0)
            angle = round(angle / SNAP) * SNAP
            v_stable = np.array([np.cos(angle), np.sin(angle), 0.0])
            grasp_Z = np.array([0.0, 0.0, 1.0])
            grasp_X = np.cross(v_stable, grasp_Z)
            if np.linalg.norm(grasp_X) < 1e-6:
                grasp_X = np.array([1.0, 0.0, 0.0])
            else:
                grasp_X /= np.linalg.norm(grasp_X)
            grasp_Y = np.cross(grasp_Z, grasp_X)

        quat = R_sci.from_matrix(np.column_stack((grasp_X, grasp_Y, grasp_Z))).as_quat()

        # Metric box trong XY metric (met) - dung cho SAT collision (cung don vi voi tool rect)
        pts_xy_m = valid_pts[:, :2].astype(np.float32)
        metric_box = cv2.boxPoints(cv2.minAreaRect(pts_xy_m)).astype(np.float32)

        return {
            "pos": [float(cx), float(cy), float(cz)],
            "quat": [float(q) for q in quat],
            "v_major": v_major.tolist(),
            "z_max": z_max,
            "dims": [L, W, H],
            "is_circular": is_circular,
            "is_symmetric": is_symmetric,
            "metric_box": metric_box,
            "oriented_box": box_pts
        }
    def _project_3d_to_2d(self, pt_3d):
        """Chiếu một điểm 3D xuống pixel 2D dùng camera intrinsics."""
        intr = self._active_intrinsics
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
                        # BƯỚC 1: AI NHẬN DIỆN
                        ai_result = self.ai.get_mask_contour(self._color_img)
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
                        cnts, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts:
                            cnt = max(cnts, key=cv2.contourArea)
                            p = self._extract_3d_pose(cnt)
                            if p:
                                processed_objs.append({
                                    "name": obj["name"],
                                    "mask": mask_bool,
                                    "contour": cnt,
                                    "pos": p["pos"],
                                    "quat": p["quat"],
                                    "v_major": p["v_major"],
                                    "z_max": p["z_max"],
                                    "dims": p["dims"],
                                    "is_circular": p["is_circular"],
                                    "is_symmetric": p["is_symmetric"],
                                    "metric_box": p["metric_box"],
                                    "oriented_box": p["oriented_box"],
                                })

                # BƯỚC 3: SẮP XẾP ƯU TIÊN
                sorted_objs = self.priority.sort(processed_objs)

                # BƯỚC 4: KIỂM TRA VA CHẠM & CHỌN MỤC TIÊU
                best_target = None
                for obj in sorted_objs:
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
                    if is_stable and self.node.lock == False:
                        c_name, c_pos, c_quat = self.stabilizer.get_confirmed()
                        self.node.task_pub.publish(String(data=json.dumps({
                            "name": c_name,
                            "pos": c_pos,
                            "quat": c_quat,
                            "dims": best_target["dims"],
                        })))
                        print(f"[STAB] >>> PUBLISHED task for {c_name}")
                    else:
                        print(f"[STAB] Not stable yet or robot is moving, not publishing.")
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

                    display[obj["mask"]] = display[obj["mask"]] * 0.2 + np.array(color) * 0.8
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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
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
