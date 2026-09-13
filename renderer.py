"""折叠屏渲染：实时桌面折叠 + 强模糊，静态后景，无黑场，尽量流畅。

- 前景：实时抓取桌面（mss），随折叠角做透视折叠 + 渐进模糊 + 渐变透明。
- 后景：BACKDROP_PATH 指定的照片（等比裁剪铺满，只模糊不变形）；没有则用模糊压暗的桌面。
- 性能：半分辨率渲染 + 盒式模糊 + 完全展开快速路径。
"""
import os
import sys

import cv2
import numpy as np
import pygame

try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

_WDA_EXCLUDEFROMCAPTURE = 0x11

_FONT_PATHS = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyh.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
)

# ================== 背景图路径（改成你的图片路径即可） ==================
# 可以是相对项目目录的文件名，也可以是完整绝对路径，例如：
#   BACKDROP_PATH = "desk_bg.png"
#   BACKDROP_PATH = "C:/Users/xxx/Pictures/my_wallpaper.jpg"
BACKDROP_PATH = "desk_bg.png"
# ====================================================================

# 背景（向外露出部分）模糊强度：相对前景模糊的比例，<1 表示背景比前景模糊更轻
BACKDROP_BLUR_SCALE = 1


def _load_font(size):
    for p in _FONT_PATHS:
        if os.path.exists(p):
            try:
                return pygame.font.Font(p, size)
            except Exception:
                continue
    return pygame.font.Font(None, size)


def _resize_fill(img, w, h):
    """等比缩放 + 居中裁剪到 w×h，铺满且不拉伸、不留黑边。"""
    ih, iw = img.shape[:2]
    if iw == 0 or ih == 0:
        return img
    scale = max(w / iw, h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (nw - w) // 2
    y = (nh - h) // 2
    return resized[y:y + h, x:x + w]


def _base_dir():
    """程序所在目录：打包成 EXE 后是 EXE 所在目录，否则是脚本所在目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


class FoldRenderer:
    def __init__(self, max_blur=260.0, feather=90.0, render_scale=0.5, fullscreen=False):
        self.width = 1280
        self.height = 720
        self.fullscreen = fullscreen
        self.render_scale = render_scale
        self.screen = None
        self.desktop = None          # 静态前景截图
        self.backdrop = None         # 静态后景（desk_bg 或模糊暗桌面）
        self.max_blur = max_blur
        self.feather = feather
        self.show_outline = False
        self._font = None
        self._sct = None
        self._mon = None
        self._dxcam = None

    # ---------- 抓屏（一次性） ----------
    def _capture_once(self):
        try:
            import mss
            with mss.mss() as sct:
                shot = sct.grab(sct.monitors[1])
            return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    def _grab(self):
        """实时抓屏（mss/BitBlt，复用上下文）。"""
        try:
            if self._sct is None:
                import mss
                self._sct = mss.mss()
                self._mon = self._sct.monitors[1]
            shot = self._sct.grab(self._mon)
            return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    def _load_backdrop(self, frame):
        path = (BACKDROP_PATH if os.path.isabs(BACKDROP_PATH)
                else os.path.join(_base_dir(), BACKDROP_PATH))
        if os.path.exists(path):
            # 用 np.fromfile + imdecode 规避中文路径下 cv2.imread 打不开的问题
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                self.backdrop = _resize_fill(img, self.width, self.height)
                print(f"[后景] 已加载 {BACKDROP_PATH}")
                return True
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 20)
        self.backdrop = (cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) * 0.5).astype(np.uint8)
        return False

    def select_backdrop(self):
        """弹出文件选择框，让用户选择本地图片作为后景。"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            # 暂时最小化 pygame 窗口，避免全屏遮挡文件选择框
            if self.screen is not None:
                pygame.display.iconify()
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="选择背景图片",
                filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")],
            )
            root.destroy()
            # 恢复 pygame 窗口
            if self.screen is not None:
                flags = pygame.FULLSCREEN if self.fullscreen else 0
                self.screen = pygame.display.set_mode((self.width, self.height), flags)
                self._exclude_from_capture()
            if not path:
                return False
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[后景] 无法读取图片: {path}")
                return False
            self.backdrop = _resize_fill(img, self.width, self.height)
            print(f"[后景] 已加载 {path}")
            return True
        except Exception as e:
            print(f"[后景] 选择失败: {e}")
            return False

    def recapture(self):
        """标定时重新截一次屏作为前景（静态），并重建后景。"""
        d = self._capture_once()
        if d is None:
            return False
        self.desktop = d
        self.height, self.width = d.shape[:2]
        self._load_backdrop(d)
        print("[前景] 已重新截屏")
        return True

    # ---------- 窗口 ----------
    def _exclude_from_capture(self):
        try:
            import ctypes
            info = pygame.display.get_wm_info()
            hwnd = info.get('window')
            if hwnd:
                ctypes.windll.user32.SetWindowDisplayAffinity(int(hwnd), _WDA_EXCLUDEFROMCAPTURE)
                return True
        except Exception:
            pass
        return False

    def ensure_window(self):
        if self.screen is None:
            pygame.init()
            if self.desktop is None:
                self.desktop = self._capture_once()
            if self.desktop is not None:
                self.height, self.width = self.desktop.shape[:2]
                if self.backdrop is None:
                    self._load_backdrop(self.desktop)
            flags = pygame.FULLSCREEN if self.fullscreen else 0
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
            pygame.display.set_caption("折叠屏模拟")
            self._exclude_from_capture()
        return self.screen

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        self._exclude_from_capture()
        print(f"[渲染] 全屏={'开' if self.fullscreen else '关'} {self.width}x{self.height}")

    # ---------- 几何 ----------
    def project_panel(self, fold_angle_deg):
        W, H = self.width, self.height
        t = float(np.clip((180.0 - fold_angle_deg) / 180.0, 0.0, 1.0))
        D = 2.0 * H
        f = D
        cx, cy = W / 2.0, float(H)

        # 透视倾斜：保持原始完整倾斜（0 → 90°），与下面的整体缩小同时进行
        a = np.radians((180.0 - fold_angle_deg) * 0.5)
        ca, sa = np.cos(a), np.sin(a)

        def proj(x, y, z):
            d = D + z
            return (cx + f * x / d, cy - f * y / d)

        tl = proj(-W / 2, H * ca, H * sa)
        tr = proj(W / 2, H * ca, H * sa)
        br = proj(W / 2, 0.0, 0.0)
        bl = proj(-W / 2, 0.0, 0.0)

        # 垂直居中：折叠时面板整体上移，保持其中心在屏幕垂直中心
        ys = [tl[1], tr[1], br[1], bl[1]]
        shift = H / 2.0 - (min(ys) + max(ys)) / 2.0
        corners = [(x, y + shift) for (x, y) in (tl, tr, br, bl)]

        # 整体渐进缩小：1.0 -> 0.7，向屏幕中心缩放
        scale = 1.0 - 0.3 * t
        ccx, ccy = W / 2.0, H / 2.0
        return [(ccx + (x - ccx) * scale, ccy + (y - ccy) * scale) for (x, y) in corners]

    # ---------- 绘制 ----------
    def draw(self, fold_angle_deg, hud=""):
        self.ensure_window()
        self.screen.fill((0, 0, 0))

        # 实时抓取前景
        d = self._grab()
        if d is not None:
            self.desktop = d

        if self.desktop is None:
            pygame.draw.polygon(self.screen, (0, 255, 0), self.project_panel(fold_angle_deg), 2)
        else:
            self._draw(fold_angle_deg)

        if self._font is None:
            self._font = _load_font(22)
        font = self._font
        hud_surf = font.render(hud, True, (0, 255, 0))
        self.screen.blit(hud_surf, (10, 10))
        hint = "SPACE=标定  O=选背景  D=调试  R=翻转  +/-=灵敏度  B=绿框  F11=全屏  ESC=退出"
        hint_surf = font.render(hint, True, (170, 170, 170))
        self.screen.blit(hint_surf, (10, self.height - 30))
        pygame.display.flip()

    def _draw(self, fold_angle_deg):
        t = float(np.clip((180.0 - fold_angle_deg) / 180.0, 0.0, 1.0))
        W, H = self.width, self.height
        live = self.desktop

        # 完全展开：直接清晰全屏（快速路径）
        if t < 0.04:
            rgb = cv2.cvtColor(live, cv2.COLOR_BGR2RGB)
            self.screen.blit(pygame.image.frombuffer(rgb.tobytes(), (W, H), "RGB"), (0, 0))
            return

        s = self.render_scale
        rw, rh = max(2, int(W * s)), max(2, int(H * s))
        work = cv2.resize(live, (rw, rh), interpolation=cv2.INTER_AREA)
        bd = cv2.resize(self.backdrop, (rw, rh), interpolation=cv2.INTER_AREA)

        # 模糊核大小（随折叠角增大）
        k = max(1, min(int(self.max_blur * t * s), rw - 1, rh - 1))

        # 背景（desk_bg）：只渐进模糊（比前景轻）、不形变
        k_bg = max(1, min(int(k * BACKDROP_BLUR_SCALE), rw - 1, rh - 1))
        if k_bg > 1:
            bd = cv2.addWeighted(bd, 1.0 - t, cv2.blur(bd, (k_bg, k_bg)), t, 0.0)

        # 前景（截图）：渐进模糊 + 透视形变
        if k > 1:
            img = cv2.addWeighted(work, 1.0 - t, cv2.blur(work, (k, k)), t, 0.0)
        else:
            img = work

        # 透视形变（仅前景）
        dst = np.float32(self.project_panel(fold_angle_deg)) * s
        src = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (rw, rh))

        # 羽化蒙版（盒式模糊 O(1)）
        mask = np.zeros((rh, rw), np.uint8)
        cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
        fk = max(1, min(int(self.feather * s), rw - 1, rh - 1))
        mask = cv2.blur(mask, (fk, fk)).astype(np.float32) / 255.0

        # 前景渐变透明：随折叠角从 0 变到约 0.7（越折叠越透出背景）
        fg_alpha = 1.0 - 0.7 * t
        # 合成到（已渐进模糊的）背景（无黑场）：out = bd + (warped - bd) * mask * fg_alpha
        alpha = mask[..., None] * fg_alpha
        bdf = bd.astype(np.float32)
        out = warped.astype(np.float32)
        out -= bdf
        out *= alpha
        out += bdf
        np.clip(out, 0, 255, out=out)
        out = out.astype(np.uint8)

        # 上采样 + 上屏
        up = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
        self.screen.blit(pygame.image.frombuffer(rgb.tobytes(), (W, H), "RGB"), (0, 0))

        if self.show_outline:
            pygame.draw.polygon(self.screen, (0, 255, 0),
                                self.project_panel(fold_angle_deg), 2)
