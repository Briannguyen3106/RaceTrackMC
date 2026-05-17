# config.py
import os

# ── Tham số môi trường ──────────────────────────────────────────
NOISE_PROB  = 0.1
MAX_SPEED   = 4
MAX_STEPS   = 500
TRACK_ID    = 1

# ── Tham số agent ───────────────────────────────────────────────
EPSILON     = 0.1
GAMMA       = 1.0

# ── Training ────────────────────────────────────────────────────
EPISODES    = 10_000

# ── Output — luôn lấy theo vị trí file config.py ───────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_HERE, 'results')