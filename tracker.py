"""角度识别核心：帧间跟踪 + 滑动关键帧 + 固定锚点校正 + 纯旋转拟合 + One Euro 滤波。

为什么这么设计：
- 帧间跟踪（滑动关键帧）：每帧只和 ~7° 前的关键帧匹配，增量累加，保证大角度下
  仍能跟踪（固定单一基准帧会因画面重叠太小而失效）。
- 纯旋转拟合：相机绕铰链旋转为主，用 3 自由度旋转比 8 自由度单应矩阵更稳、无多解。
- 固定锚点校正：增量累加会随帧数随机游走漂移（合上再打开后回不到原位）。
  于是维护一组「固定锚点」——标定时建 home 锚点（角度 0），转动中每隔 ~12° 自动
  新建一个锚点（记住当时的画面和角度）。每帧就近匹配锚点，用「锚点角度 + 相对旋转」
  得到绝对角来校正累计角，从而把漂移钉住、回到 home 时拉回真实 0°。
- One Euro 滤波：慢速强平滑抑制抖动，快速保留响应。
- CLAHE：局部对比度归一化，抵抗自动曝光漂移。
"""
import time

import cv2
import numpy as np


class OneEuroFilter:
    """一欧拉滤波：自适应低通，慢速平滑、快速跟手。"""

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, dt):
        if self.x_prev is None:
            self.x_prev = float(x)
            self.t_prev = dt
            return float(x)
        dt = max(dt, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = float(x_hat)
        self.dx_prev = float(dx_hat)
        self.t_prev = dt
        return float(x_hat)


# 常量
REKEY_RAD = 0.12          # 自关键帧累计旋转超过 ~7° 就换关键帧
MAX_STEP_RAD = 0.30       # 单帧增量超过 ~17° 视为误匹配，丢弃
MIN_MATCHES = 12          # 最少匹配数
ANCHOR_SPACING = 0.21     # 锚点间距 ~12°
ANCHOR_MATCH_RANGE = 0.44  # 在累计角 ±25° 内找锚点
ANCHOR_MIN_INLIERS = 15   # 锚点匹配最少内点数


class AngleTracker:
    def __init__(self, camera_index=0, with_camera=True, nfeatures=1200, ratio=0.75):
        self.cap = None
        if with_camera:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"无法打开摄像头 index={camera_index}。"
                    "请确认摄像头未被占用，或改用其它 index（修改 main.py）。"
                )
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            except Exception:
                pass

        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ratio = ratio
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 滑动关键帧
        self.ref_gray = None
        self.ref_kp = None
        self.ref_des = None
        self.K = None

        self.total_rad = 0.0       # 相对标定基准、到当前关键帧的累计俯仰角
        self.last_dtheta = 0.0     # 最近一次增量旋转（调试用）

        # 固定锚点：每个是 {'gray','kp','des','angle'}
        self.anchors = []
        self._frame_count = 0

        self.oneuro = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
        self._last_t = None

        # 调试状态
        self.last_kp = None
        self.last_good = []
        self.last_src = None
        self.last_dst = None
        self.dbg_ref_gray = None
        self.dbg_ref_kp = None
        self.dbg_cur_gray = None
        self.dbg_cur_kp = None
        self.dbg_matches = None

    # ---------- 标定 ----------
    def set_reference(self, gray):
        """把当前画面设为标定基准（完全展开、无变形），清零累计角与锚点。"""
        gray = self.clahe.apply(gray)
        self._set_keyframe(gray, None, None)
        self.total_rad = 0.0
        self.oneuro.reset()
        # 重置锚点：只保留 home 锚点（绝对角度 0）
        self.anchors = [{'gray': self.ref_gray, 'kp': self.ref_kp,
                         'des': self.ref_des, 'angle': 0.0}]
        n = len(self.ref_kp) if self.ref_kp is not None else 0
        return n > 20

    def _set_keyframe(self, gray, kp, des):
        self.ref_gray = gray.copy()
        if kp is None or des is None:
            kp, des = self.orb.detectAndCompute(gray, None)
        self.ref_kp = kp
        self.ref_des = des
        if self.K is None and gray is not None:
            h, w = gray.shape[:2]
            f = float(max(h, w))
            self.K = np.array([[f, 0, w / 2.0],
                               [0, f, h / 2.0],
                               [0, 0, 1.0]], dtype=np.float64)

    def _add_anchor(self, gray, kp, des, angle):
        self.anchors.append({'gray': gray.copy(), 'kp': kp, 'des': des, 'angle': angle})
        self.anchors.sort(key=lambda a: a['angle'])

    @property
    def has_reference(self):
        return self.ref_des is not None

    def read(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if (ok and frame is not None) else None

    # ---------- 匹配 + 旋转 ----------
    def _match_and_rotate(self, desc1, kp1, desc2, kp2):
        """匹配两组特征并用纯旋转拟合。返回 (俯仰角弧度, 内点数, 匹配列表, src, dst)。"""
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance:
                good.append(pair[0])
        if len(good) < MIN_MATCHES:
            return None, len(good), good, None, None

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        if H is None:
            return None, 0, good, src.reshape(-1, 2), dst.reshape(-1, 2)

        inl = mask.ravel() == 1
        R = self._rotation_from_matches(src[inl].reshape(-1, 2), dst[inl].reshape(-1, 2))
        if R is None:
            return None, 0, good, src.reshape(-1, 2), dst.reshape(-1, 2)

        rvec, _ = cv2.Rodrigues(R)
        return float(rvec[0][0]), int(inl.sum()), good, src.reshape(-1, 2), dst.reshape(-1, 2)

    def _rotation_from_matches(self, src, dst):
        """用纯旋转模型（Wahba / SVD）拟合内点，返回旋转矩阵 R 或 None。"""
        if len(src) < 6:
            return None
        Kinv = np.linalg.inv(self.K)
        ones = np.ones((len(src), 1))
        p1 = (Kinv @ np.hstack([src, ones]).T).T
        p2 = (Kinv @ np.hstack([dst, ones]).T).T
        p1 = p1 / np.linalg.norm(p1, axis=1, keepdims=True)
        p2 = p2 / np.linalg.norm(p2, axis=1, keepdims=True)

        B = p2.T @ p1
        U, _, Vt = np.linalg.svd(B)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        return R

    # ---------- 锚点校正 ----------
    def _anchor_correction(self, kp, des, current_angle):
        """就近匹配固定锚点，返回校正后的绝对角；无可靠锚点返回 None。"""
        if not self.anchors:
            return None
        nearby = [a for a in self.anchors
                  if abs(a['angle'] - current_angle) < ANCHOR_MATCH_RANGE]
        if not nearby:
            return None
        nearby.sort(key=lambda a: abs(a['angle'] - current_angle))

        best_angle = None
        best_inl = -1
        for a in nearby[:2]:
            d, inl, _, _, _ = self._match_and_rotate(a['des'], a['kp'], des, kp)
            if d is not None and inl > best_inl:
                best_inl = inl
                best_angle = a['angle'] + d
        if best_angle is None or best_inl < ANCHOR_MIN_INLIERS:
            return None
        return best_angle

    # ---------- 估计 ----------
    def estimate(self, gray, dt=None):
        """返回滤波后的绝对俯仰角（弧度）；无法估计时返回 None。"""
        self.last_kp = None
        self.last_good = []
        self.last_src = None
        self.last_dst = None

        if not self.has_reference:
            return None

        gray = self.clahe.apply(gray)
        kp, des = self.orb.detectAndCompute(gray, None)
        self.last_kp = kp
        if des is None or len(kp) < 10:
            return None

        dtheta, inliers, good, src, dst = self._match_and_rotate(
            self.ref_des, self.ref_kp, des, kp)
        self.last_good = good
        self.last_src = src
        self.last_dst = dst
        self.last_dtheta = dtheta if dtheta is not None else 0.0

        # 调试快照
        self.dbg_ref_gray = self.ref_gray
        self.dbg_ref_kp = self.ref_kp
        self.dbg_cur_gray = gray
        self.dbg_cur_kp = kp
        self.dbg_matches = good

        if dtheta is None or len(good) < MIN_MATCHES:
            self._set_keyframe(gray, kp, des)
            return None
        if abs(dtheta) > MAX_STEP_RAD:
            self._set_keyframe(gray, kp, des)
            return None

        current_angle = self.total_rad + dtheta

        # 固定锚点校正（消除漂移）：每 3 帧做一次，软校正避免跳变顿挫
        self._frame_count += 1
        corrected = None
        if self._frame_count % 5 == 0:
            corrected = self._anchor_correction(kp, des, current_angle)

        if corrected is not None:
            # 直接用锚点绝对角（无漂移），关键帧同步为当前帧；
            # 跳变仅等于漂移量（很小），One Euro 滤波会平滑掉
            self.total_rad = corrected
            self._set_keyframe(gray, kp, des)
            out_angle = corrected
        else:
            out_angle = current_angle
            if abs(dtheta) > REKEY_RAD:
                self.total_rad = current_angle
                self._set_keyframe(gray, kp, des)

        # 距离已有锚点足够远时，新建一个固定锚点
        if self.anchors:
            nearest = min(abs(a['angle'] - out_angle) for a in self.anchors)
        else:
            nearest = float('inf')
        if nearest > ANCHOR_SPACING:
            self._add_anchor(gray, kp, des, out_angle)

        if dt is None:
            now = time.monotonic()
            dt = now - self._last_t if self._last_t is not None else 1.0 / 30.0
            self._last_t = now
        dt = float(np.clip(dt, 1e-3, 0.2))
        return self.oneuro.filter(out_angle, dt)

    # ---------- 调试 ----------
    def draw_matches(self, cur_gray=None):
        if (self.dbg_ref_gray is None or self.dbg_ref_kp is None
                or self.dbg_cur_gray is None or self.dbg_cur_kp is None
                or not self.dbg_matches):
            fallback = cur_gray if cur_gray is not None else self.dbg_cur_gray
            if fallback is None:
                return np.zeros((240, 320, 3), dtype=np.uint8)
            return cv2.cvtColor(fallback, cv2.COLOR_GRAY2BGR)
        return cv2.drawMatches(
            self.dbg_ref_gray, self.dbg_ref_kp, self.dbg_cur_gray, self.dbg_cur_kp,
            self.dbg_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
