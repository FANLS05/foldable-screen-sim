"""折叠屏模拟 - 阶段二：桌面截图折叠 + 磨砂模糊 + 3D 形变（含计时日志）。

运行：
    python main.py

控制：
    SPACE  标定基准帧 + 重新截屏当前景
    D      开关特征匹配调试窗口
    R      翻转角度符号
    +/-    调整灵敏度 SCALE
    B      开关绿色轮廓
    F11    切换全屏
    ESC/Q  退出（退出时把每帧耗时写入 timing_log.csv）
"""
import csv
import time

import cv2
import numpy as np
import pygame

from tracker import AngleTracker
from renderer import FoldRenderer


def main():
    tracker = AngleTracker(camera_index=0)
    renderer = FoldRenderer(fullscreen=True)
    renderer.ensure_window()

    SCALE = 1.1     # 物理俯仰角 -> 虚拟折叠角 的比例
    SIGN = -1       # 让"合上上盖 -> 折叠角减小"；方向反了按 R
    fold_angle = 180.0
    show_debug = False
    last_gray = None
    pitch_deg = 0.0
    fps = 0.0

    # 计时日志：每帧 (frame, t_sec, frame_ms, cam_ms, track_ms, render_ms,
    #           pitch_deg, fold_angle, matches, fps) 
    timing = []
    frame_idx = 0
    t_start = time.perf_counter()
    prev_loop = time.perf_counter()

    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            loop = time.perf_counter()
            frame_ms = (loop - prev_loop) * 1000.0
            prev_loop = loop

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif k == pygame.K_SPACE:
                        renderer.recapture()  # 标定同时重新截一次屏当前景（静态）
                        if last_gray is not None:
                            ok = tracker.set_reference(last_gray)
                            print(f"[标定] 基准帧 {'成功' if ok else '失败(特征点不足)'}")
                    elif k == pygame.K_d:
                        show_debug = not show_debug
                    elif k == pygame.K_b:
                        renderer.show_outline = not renderer.show_outline
                    elif k == pygame.K_r:
                        SIGN = -SIGN
                        print(f"[参数] SIGN = {SIGN:+d}")
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS):
                        SCALE = round(SCALE + 0.1, 2)
                        print(f"[参数] SCALE = {SCALE}")
                    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        SCALE = round(max(0.1, SCALE - 0.1), 2)
                        print(f"[参数] SCALE = {SCALE}")
                    elif k == pygame.K_F11:
                        renderer.toggle_fullscreen()

            # 相机读取 + 灰度
            t0 = time.perf_counter()
            frame = tracker.read()
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_gray = gray
            t1 = time.perf_counter()
            cam_ms = (t1 - t0) * 1000.0
            if frame is None:
                continue

            # 角度识别
            t2 = time.perf_counter()
            pitch_rad = tracker.estimate(gray)
            t3 = time.perf_counter()
            track_ms = (t3 - t2) * 1000.0
            match_count = len(tracker.last_good)

            if pitch_rad is not None:
                pitch_deg = float(np.degrees(pitch_rad))
                fold_angle = float(np.clip(180.0 + SIGN * SCALE * pitch_deg, 0.0, 180.0))

            if pitch_rad is not None:
                status = "追踪中"
            elif not tracker.has_reference:
                status = "未标定(按SPACE)"
            else:
                status = "匹配不足"
            hud = (f"fold={fold_angle:5.1f}  pitch={pitch_deg:+6.2f}°  "
                   f"matches={match_count}  FPS={fps:4.1f}  SCALE={SCALE}  SIGN={SIGN:+d}  {status}")

            # 渲染
            t4 = time.perf_counter()
            renderer.draw(fold_angle, hud)
            t5 = time.perf_counter()
            render_ms = (t5 - t4) * 1000.0

            if show_debug:
                dbg = tracker.draw_matches(gray)
                cv2.imshow("debug (D 关闭)", dbg)

            clock.tick(30)
            fps = 0.9 * fps + 0.1 * clock.get_fps()

            timing.append((frame_idx, round(time.perf_counter() - t_start, 3),
                           round(frame_ms, 2), round(cam_ms, 2), round(track_ms, 2),
                           round(render_ms, 2), round(pitch_deg, 2), round(fold_angle, 1),
                           match_count, round(fps, 1)))
            frame_idx += 1
    finally:
        tracker.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        _write_log(timing)


def _write_log(timing):
    with open("timing_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_sec", "frame_ms", "cam_ms", "track_ms", "render_ms",
                    "pitch_deg", "fold_angle", "matches", "fps"])
        w.writerows(timing)
    print(f"[日志] 已保存 timing_log.csv，共 {len(timing)} 帧")


if __name__ == "__main__":
    main()
