import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
# track.py
import numpy as np
from config import MAX_SPEED, NOISE_PROB, MAX_STEPS

# ── Định nghĩa 2 bản đồ ─────────────────────────────────────────────────────
# Mỗi bản đồ là list các string, mỗi ký tự là 1 ô:
#   '.' = đường đua     'W' = tường
#   'S' = vạch xuất phát  'F' = vạch đích

TRACK_1 = [
    "WWWWWWWWWWWWWWWWWWW",
    "WWWW.............FW",
    "WWW..............FW",
    "WWW..............FW",
    "WW...............FW",
    "W................FW",
    "W................FW",
    "W..........WWWWWWWW",
    "W.........WWWWWWWWW",
    "W.........WWWWWWWWW",
    "W.........WWWWWWWWW",
    "W.........WWWWWWWWW",
    "W.........WWWWWWWWW",
    "W.........WWWWWWWWW",
    "W.........WWWWWWWWW",
    "WW........WWWWWWWWW",
    "WW........WWWWWWWWW",
    "WW........WWWWWWWWW",
    "WW........WWWWWWWWW",
    "WW........WWWWWWWWW",
    "WW........WWWWWWWWW",
    "WW........WWWWWWWWW",
    "WW........WWWWWWWWW",
    "WWW.......WWWWWWWWW",
    "WWW.......WWWWWWWWW",
    "WWW.......WWWWWWWWW",
    "WWW.......WWWWWWWWW",
    "WWW.......WWWWWWWWW",
    "WWW.......WWWWWWWWW",
    "WWW.......WWWWWWWWW",
    "WWWW......WWWWWWWWW",
    "WWWW......WWWWWWWWW",
    "WWWWSSSSSSWWWWWWWWW",
    "WWWWWWWWWWWWWWWWWWW",
]

TRACK_2 = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    "WWWWWWWWWWWWWWWWW...............FW",
    "WWWWWWWWWWWWWW..................FW",
    "WWWWWWWWWWWWW...................FW",
    "WWWWWWWWWWWW....................FW",
    "WWWWWWWWWWWW....................FW",
    "WWWWWWWWWWWW....................FW",
    "WWWWWWWWWWWW....................FW",
    "WWWWWWWWWWWWW...................FW",
    "WWWWWWWWWWWWWW..................FW",
    "WWWWWWWWWWWWWWW................WWW",
    "WWWWWWWWWWWWWWW.............WWWWWW",
    "WWWWWWWWWWWWWWW............WWWWWWW",
    "WWWWWWWWWWWWWWW..........WWWWWWWWW",
    "WWWWWWWWWWWWWWW.........WWWWWWWWWW",
    "WWWWWWWWWWWWWW..........WWWWWWWWWW",
    "WWWWWWWWWWWWW...........WWWWWWWWWW",
    "WWWWWWWWWWWW............WWWWWWWWWW",
    "WWWWWWWWWWW.............WWWWWWWWWW",
    "WWWWWWWWWW..............WWWWWWWWWW",
    "WWWWWWWWW...............WWWWWWWWWW",
    "WWWWWWWW................WWWWWWWWWW",
    "WWWWWWW.................WWWWWWWWWW",
    "WWWWWW..................WWWWWWWWWW",
    "WWWWW...................WWWWWWWWWW",
    "WWWW....................WWWWWWWWWW",
    "WWW.....................WWWWWWWWWW",
    "WW......................WWWWWWWWWW",
    "W.......................WWWWWWWWWW",
    "W.......................WWWWWWWWWW",
    "WSSSSSSSSSSSSSSSSSSSSSSSWWWWWWWWWW",
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
]

TRACKS = {1: TRACK_1, 2: TRACK_2}


class RaceTrack:
    """Môi trường Racetrack theo Sutton & Barto Example 5.8."""

    def __init__(self, track_id: int):
        raw = TRACKS[track_id]

        # Lưu kích thước
        self.n_rows = len(raw)
        self.n_cols = len(raw[0])

        # Chuyển sang numpy array 2D cho dễ index
        self.grid = np.array([list(row) for row in raw])

        # Thu thập tọa độ vạch xuất phát và đích
        self.start_cells  = self._find_cells('S')
        self.finish_cells = self._find_cells('F')

        # Không gian vận tốc: 0..MAX_SPEED mỗi chiều
        self.v_range = range(0, MAX_SPEED + 1)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _find_cells(self, symbol: str):
        """Trả về list các (row, col) có ký tự = symbol."""
        positions = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r, c] == symbol:
                    positions.append((r, c))
        return positions

    def _is_road(self, r: int, c: int) -> bool:
        """Ô (r,c) có thể đi được không (road, start, finish)?"""
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return self.grid[r, c] in ('.', 'S', 'F')

    def _is_finish(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return False
        return self.grid[r, c] == 'F'

    def _random_start(self):
        """Chọn ngẫu nhiên 1 ô trên vạch xuất phát."""
        idx = np.random.randint(len(self.start_cells))
        return self.start_cells[idx]

    # ── API chính ────────────────────────────────────────────────────────────

    def reset(self):
        """
        Bắt đầu episode mới.
        Trả về state = (row, col, vr, vc) — vận tốc khởi đầu = (0, 0).
        """
        r, c = self._random_start()
        return (r, c, 0, 0)

    def step(self, state: tuple, action: tuple, use_noise: bool = True):
        """
        Thực hiện 1 bước di chuyển.

        Tham số
        -------
        state      : (row, col, vr, vc)
        action     : (delta_vr, delta_vc) — mỗi giá trị trong {-1, 0, +1}
        use_noise  : nếu True, 10% xác suất action bị triệt tiêu

        Trả về
        ------
        next_state : (row, col, vr, vc)  — None nếu về đích
        reward     : luôn = -1
        done       : True nếu xe qua finish line
        hit_wall   : True nếu va tường (đã reset về start)
        """
        r, c, vr, vc = state
        dvr, dvc     = action

        # ── 1. Áp noise ───────────────────────────────────────────────────
        if use_noise and np.random.random() < NOISE_PROB:
            dvr, dvc = 0, 0

        # ── 2. Cập nhật vận tốc ───────────────────────────────────────────
        new_vr = int(np.clip(vr + dvr, 0, MAX_SPEED))
        new_vc = int(np.clip(vc + dvc, 0, MAX_SPEED))

        # Không cho cả hai = 0 (trừ vạch xuất phát)
        if new_vr == 0 and new_vc == 0:
            new_vr, new_vc = vr, vc   # giữ nguyên vận tốc cũ

         # ── 3. Kiểm tra điểm cuối ─────────────────────────────────────────
        nr = r - new_vr
        nc = c + new_vc
    
        if self._is_finish(nr, nc):
            return None, -1, True, False
    
        if not self._is_road(nr, nc):
            start_r, start_c = self._random_start()
            return (start_r, start_c, 0, 0), -1, False, True
    
        return (nr, nc, new_vr, new_vc), -1, False, False