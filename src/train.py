# train.py
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import numpy as np
from config import EPISODES, TRACK_ID
from src.track import RaceTrack
from src.agents import OnPolicyAgent, WISAgent, OISAgent, PerDecisionAgent, ACTIONS


def train(agent, track, n_episodes: int = EPISODES,
          log_every: int = 500, use_noise: bool = True) -> dict:
    """
    Chạy vòng lặp training.

    Tham số
    -------
    agent       : instance của MCAgent (bất kỳ subclass nào)
    track       : instance của RaceTrack
    n_episodes  : số episodes huấn luyện
    log_every   : in log mỗi bao nhiêu episodes
    use_noise   : True = có nhiễu 10% (đúng đề bài), False = không nhiễu

    Trả về
    ------
    dict gồm:
        episode_lengths : list độ dài từng episode
        rewards         : list tổng reward từng episode (= -length)
        success_rates   : list success rate theo từng cửa sổ log_every
    """
    episode_lengths = []
    rewards         = []
    success_rates   = []

    for ep in range(1, n_episodes + 1):
        length = agent.train_episode(track, use_noise=use_noise)
        episode_lengths.append(length)
        rewards.append(-length)

        if ep % log_every == 0:
            window       = episode_lengths[-log_every:]
            avg_length   = np.mean(window)
            success_rate = np.mean([1 if l < 500 else 0 for l in window])
            success_rates.append(success_rate)
            print(
                f"[{ep:>6}/{n_episodes}] "
                f"avg length: {avg_length:6.1f} | "
                f"success rate: {success_rate*100:5.1f}%"
            )

    return {
        "episode_lengths": episode_lengths,
        "rewards":         rewards,
        "success_rates":   success_rates,
        'agent': agent
    }


def run_demo(agent, track, n_demos: int = 5) -> list:
    """
    Chạy n_demos trajectory theo greedy policy, tắt noise.
    Trả về list các trajectory để plot.py vẽ.
    """
    demos = []
    for i in range(n_demos):
        trajectory = []
        state      = track.reset()

        for _ in range(500):
            action_idx = agent.greedy_action(state)
            action     = ACTIONS[action_idx]
            next_state, _, done, _ = track.step(state, action, use_noise=False)
            trajectory.append(state)
            if done:
                break
            state = next_state

        demos.append(trajectory)
        print(f"  Demo {i+1}: {len(trajectory)} bước")

    return demos


def compare_agents(track, n_episodes: int = EPISODES,
                   use_noise: bool = True) -> dict:
    """
    Train tất cả agents trên cùng track, trả về kết quả để so sánh.
    """
    agents = {
        "On-policy":     OnPolicyAgent(),
        "WIS (random)":  WISAgent(behaviour='random'),
        "WIS (epsilon)": WISAgent(behaviour='epsilon'),
        "OIS (random)":  OISAgent(behaviour='random'),
        "OIS (epsilon)": OISAgent(behaviour='epsilon'),
        "Per-decision":  PerDecisionAgent(),
    }

    results = {}
    for name, agent in agents.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}  |  noise={'ON' if use_noise else 'OFF'}")
        print('='*50)
        results[name] = train(agent, track, n_episodes, use_noise=use_noise)
        results[name]['agent'] = agent

    return results