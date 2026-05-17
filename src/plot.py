import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
# plot.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from config import RESULTS_DIR

# Màu cho từng loại ô trên track
CELL_COLORS = {
    'W': '#D3D1C7',  # tường
    '.': '#F1EFE8',  # đường đua
    'S': '#F0997B',  # vạch xuất phát
    'F': '#5DCAA5',  # vạch đích
}

# Màu cho từng agent khi vẽ so sánh
AGENT_COLORS = {
    'On-policy':     '#185FA5',
    'WIS (random)':  '#E24B4A',
    'WIS (epsilon)': '#F0997B',
    'OIS (random)':  '#1D9E75',
    'OIS (epsilon)': '#5DCAA5',
    'Per-decision':  '#9B59B6',
}


def _ensure_dir():
    """Tạo thư mục results nếu chưa có."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _draw_track(ax, track):
    """Vẽ grid bản đồ lên axes ax."""
    for r in range(track.n_rows):
        for c in range(track.n_cols):
            cell = track.grid[r, c]
            color = CELL_COLORS.get(cell, '#FFFFFF')
            ax.add_patch(plt.Rectangle(
                (c, track.n_rows - r - 1),  # lật trục y để row 0 ở trên
                1, 1,
                facecolor=color,
                edgecolor='#CCCCCC',
                linewidth=0.3,
            ))
    ax.set_xlim(0, track.n_cols)
    ax.set_ylim(0, track.n_rows)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_track(track, title: str = "Racetrack", save: bool = True):
    """Vẽ bản đồ track với chú thích."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 8))
    _draw_track(ax, track)

    # Legend
    patches = [
        mpatches.Patch(color=CELL_COLORS['S'], label='Xuất phát'),
        mpatches.Patch(color=CELL_COLORS['F'], label='Đích'),
        mpatches.Patch(color=CELL_COLORS['.'], label='Đường đua'),
        mpatches.Patch(color=CELL_COLORS['W'], label='Tường'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8)
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'track.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu: {path}")
    plt.show()


def plot_trajectory(track, trajectories: list, title: str = "Trajectory", save: bool = True):
    """
    Vẽ các trajectory lên track.

    trajectories : list of list of state (mỗi state là (r, c, vr, vc))
    """
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 8))
    _draw_track(ax, track)

    colors = ['#185FA5', '#E24B4A', '#1D9E75', '#9B59B6', '#F0997B']
    for i, traj in enumerate(trajectories):
        color = colors[i % len(colors)]
        cols  = [s[1] + 0.5 for s in traj]           # +0.5 để vẽ giữa ô
        rows  = [track.n_rows - s[0] - 0.5 for s in traj]  # lật trục y

        ax.plot(cols, rows, '-o',
                color=color, linewidth=1.5,
                markersize=3, alpha=0.8,
                label=f'Demo {i+1} ({len(traj)} bước)')

        # Đánh dấu điểm xuất phát
        ax.plot(cols[0], rows[0], 's',
                color=color, markersize=7, zorder=5)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'trajectory.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu: {path}")
    plt.show()


def plot_learning_curve(results: dict, window: int = 200, save: bool = True):
    """
    Vẽ learning curve (reward trung bình theo episodes) cho tất cả agents.

    results : dict trả về từ compare_agents()
    window  : cửa sổ moving average
    """
    _ensure_dir()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, data in results.items():
        color   = AGENT_COLORS.get(name, '#333333')
        rewards = data['rewards']

        # Moving average
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        x  = np.arange(window, len(rewards) + 1)

        ax1.plot(x, ma, label=name, color=color, linewidth=1.5, alpha=0.9)

    ax1.set_ylabel('Avg reward (moving avg)', fontsize=10)
    ax1.set_title('Learning Curve — Reward', fontsize=12)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    # Success rate
    for name, data in results.items():
        color   = AGENT_COLORS.get(name, '#333333')
        lengths = data['episode_lengths']
        success = [1 if l < 500 else 0 for l in lengths]
        ma      = np.convolve(success, np.ones(window) / window, mode='valid')
        x       = np.arange(window, len(success) + 1)
        ax2.plot(x, ma * 100, label=name, color=color, linewidth=1.5, alpha=0.9)

    ax2.set_xlabel('Episodes', fontsize=10)
    ax2.set_ylabel('Success rate (%)', fontsize=10)
    ax2.set_title('Learning Curve — Success Rate', fontsize=12)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'learning_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu: {path}")
    plt.show()


def plot_episode_length_distribution(results: dict, save: bool = True):
    """
    Vẽ histogram phân phối độ dài episode cho từng agent
    (chỉ tính các episode thành công).
    """
    _ensure_dir()
    n     = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes  = axes.flatten()

    for idx, (name, data) in enumerate(results.items()):
        ax      = axes[idx]
        color   = AGENT_COLORS.get(name, '#333333')
        lengths = [l for l in data['episode_lengths'] if l < 500]

        if lengths:
            ax.hist(lengths, bins=40, color=color, alpha=0.8, edgecolor='white')
            ax.axvline(np.mean(lengths), color='black',
                       linewidth=1.5, linestyle='--',
                       label=f'Mean: {np.mean(lengths):.1f}')
            ax.legend(fontsize=8)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Episode length', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Ẩn subplot thừa nếu có
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Phân phối độ dài episode (episodes thành công)', fontsize=12)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu: {path}")
    plt.show()


