import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
# agents.py
import numpy as np
from collections import defaultdict
from config import EPSILON, GAMMA, MAX_SPEED

# 9 actions: tổ hợp (delta_vr, delta_vc) trong {-1, 0, +1} x {-1, 0, +1}
ACTIONS = [(dvr, dvc) for dvr in [-1, 0, 1] for dvc in [-1, 0, 1]]


def valid_actions(vr: int, vc: int) -> list:
    """Lọc các action hợp lệ — kết quả không được làm cả hai vận tốc = 0."""
    result = []
    for i, (dvr, dvc) in enumerate(ACTIONS):
        new_vr = int(np.clip(vr + dvr, 0, MAX_SPEED))
        new_vc = int(np.clip(vc + dvc, 0, MAX_SPEED))
        if new_vr == 0 and new_vc == 0:
            continue
        result.append(i)
    return result if result else [4]  # action (0,0) làm fallback


# ── Class base ───────────────────────────────────────────────────────────────

class MCAgent:
    """
    Base class cho tất cả thuật toán Monte Carlo.
    Subclass chỉ cần override _update().
    """

    def __init__(self, epsilon: float = EPSILON, gamma: float = GAMMA):
        self.epsilon = epsilon
        self.gamma   = gamma

        # Q[state][action_idx] = giá trị ước lượng
        self.Q = defaultdict(lambda: np.zeros(9))

        # C[state][action_idx] = denominator để tính trung bình
        # (on-policy: đếm lần thăm / off-policy: tổng trọng số)
        self.C = defaultdict(lambda: np.zeros(9))

    # ── Chọn action ──────────────────────────────────────────────────────────

    def greedy_action(self, state: tuple) -> int:
        """Chọn action có Q cao nhất trong số các action hợp lệ."""
        _, _, vr, vc = state
        va = valid_actions(vr, vc)
        q_vals = self.Q[state]
        return max(va, key=lambda i: q_vals[i])

    def epsilon_greedy(self, state: tuple) -> int:
        """Epsilon-greedy: explore với xác suất epsilon, exploit còn lại."""
        _, _, vr, vc = state
        va = valid_actions(vr, vc)
        if np.random.random() < self.epsilon:
            return np.random.choice(va)
        return self.greedy_action(state)

    def random_action(self, state: tuple) -> int:
        """Uniform random — dùng làm behaviour policy cho off-policy."""
        _, _, vr, vc = state
        va = valid_actions(vr, vc)
        return np.random.choice(va)

    # ── Sinh episode ─────────────────────────────────────────────────────────

    def generate_episode(self, track, use_noise: bool = True) -> list:
        """
        Chạy 1 episode, trả về list các (state, action_idx, reward).
        """
        trajectory = []
        state = track.reset()

        for _ in range(500):
            action_idx = self._select_action(state)
            action     = ACTIONS[action_idx]
            next_state, reward, done, _ = track.step(state, action, use_noise)
            trajectory.append((state, action_idx, reward))
            if done:
                break
            state = next_state

        return trajectory

    def _select_action(self, state: tuple) -> int:
        """Policy dùng khi generate episode — subclass override nếu cần."""
        return self.epsilon_greedy(state)

    # ── Update Q ─────────────────────────────────────────────────────────────

    def _update(self, trajectory: list):
        """Subclass bắt buộc override hàm này."""
        raise NotImplementedError

    def train_episode(self, track, use_noise: bool = True) -> int:
        """Chạy 1 episode và cập nhật Q. Trả về độ dài episode."""
        trajectory = self.generate_episode(track, use_noise=use_noise)
        self._update(trajectory)
        return len(trajectory)


# ── On-policy ε-soft ─────────────────────────────────────────────────────────

class OnPolicyAgent(MCAgent):
    """
    First-visit MC Control với epsilon-greedy.
    Cùng 1 policy vừa sinh data vừa được cải thiện.
    """

    def _update(self, trajectory: list):
        # Tính G cho từng bước trước (duyệt xuôi)
        T = len(trajectory)
        G = 0.0
        returns = []
        for _, _, reward in reversed(trajectory):
            G = self.gamma * G + reward
            returns.append(G)
        returns.reverse()  # returns[t] = G từ bước t trở đi

        visited = set()
        for t, (state, action_idx, _) in enumerate(trajectory):
            if (state, action_idx) not in visited:  # first-visit
                visited.add((state, action_idx))
                self.C[state][action_idx] += 1
                # Incremental mean: Q = Q + (G - Q) / count
                self.Q[state][action_idx] += (
                    (returns[t] - self.Q[state][action_idx])
                    / self.C[state][action_idx]
                )


# ── Off-policy base ───────────────────────────────────────────────────────────

class OffPolicyAgent(MCAgent):
    """
    Base cho WIS, OIS, PerDecision.
    Behaviour policy: 'random' (uniform) hoặc 'epsilon' (epsilon-greedy).
    Target policy: greedy.
    """

    def __init__(self, behaviour: str = 'random', **kwargs):
        super().__init__(**kwargs)
        assert behaviour in ('random', 'epsilon'), \
            "behaviour phải là 'random' hoặc 'epsilon'"
        self.behaviour = behaviour

    def _select_action(self, state: tuple) -> int:
        if self.behaviour == 'epsilon':
            return self.epsilon_greedy(state)
        return self.random_action(state)

    def _behaviour_prob(self, state: tuple, action_idx: int) -> float:
        """Xác suất của behaviour policy chọn action này."""
        _, _, vr, vc = state
        va = valid_actions(vr, vc)
        if self.behaviour == 'epsilon':
            best = self.greedy_action(state)
            if action_idx == best:
                return 1 - self.epsilon + self.epsilon / len(va)
            else:
                return self.epsilon / len(va)
        return 1.0 / len(va)  # uniform random

    def _target_prob(self, state: tuple, action_idx: int,
                     greedy_snapshot: dict) -> float:
        """
        Xác suất của target policy (greedy) chọn action này.
        Dùng greedy_snapshot thay vì gọi greedy_action() trực tiếp
        để tránh bị ảnh hưởng bởi các update Q giữa chừng.
        """
        best = greedy_snapshot.get(state, self.greedy_action(state))
        return 1.0 if action_idx == best else 0.0


# ── Off-policy WIS ────────────────────────────────────────────────────────────

class WISAgent(OffPolicyAgent):
    """
    Weighted Importance Sampling.
    Variance thấp hơn OIS, nhưng slightly biased.
    """

    def _update(self, trajectory: list):
        # Snapshot greedy action TRƯỚC khi update Q
        # → tránh target policy bị thay đổi giữa chừng
        greedy_snapshot = {
            state: self.greedy_action(state)
            for state, _, _ in trajectory
        }

        G = 0.0
        W = 1.0

        for state, action_idx, reward in reversed(trajectory):
            G = self.gamma * G + reward

            # Weighted IS update
            self.C[state][action_idx] += W
            self.Q[state][action_idx] += (
                W / self.C[state][action_idx]
            ) * (G - self.Q[state][action_idx])

            # Nếu target policy không chọn action này → dừng
            if action_idx != greedy_snapshot[state]:
                break

            b_prob = self._behaviour_prob(state, action_idx)
            W *= 1.0 / b_prob  # target prob = 1.0 nên chỉ chia b_prob

            if W > 1e6:  # tránh W bùng nổ
                break


# ── Off-policy OIS ────────────────────────────────────────────────────────────

class OISAgent(OffPolicyAgent):
    """
    Ordinary Importance Sampling.
    Unbiased nhưng variance cao hơn WIS.
    """

    def _update(self, trajectory: list):
        # Snapshot greedy action TRƯỚC khi update Q
        greedy_snapshot = {
            state: self.greedy_action(state)
            for state, _, _ in trajectory
        }

        G = 0.0
        W = 1.0

        for state, action_idx, reward in reversed(trajectory):
            G = self.gamma * G + reward

            # OIS: incremental mean đơn giản, nhân G với W
            self.C[state][action_idx] += 1
            self.Q[state][action_idx] += (
                (W * G - self.Q[state][action_idx])
                / self.C[state][action_idx]
            )

            if action_idx != greedy_snapshot[state]:
                break

            b_prob = self._behaviour_prob(state, action_idx)
            W *= 1.0 / b_prob

            if W > 1e6:
                break


# ── Per-decision IS ───────────────────────────────────────────────────────────

class PerDecisionAgent(OffPolicyAgent):
    """
    Per-decision Importance Sampling.
    Áp dụng IS ratio độc lập từng bước thay vì cả episode.
    Fix: cộng reward TRƯỚC khi nhân rho cho bước tiếp theo
    → tránh rho=0 làm mất toàn bộ contribution của reward hiện tại.
    """

    def _update(self, trajectory: list):
        # Snapshot greedy action TRƯỚC khi update Q
        greedy_snapshot = {
            state: self.greedy_action(state)
            for state, _, _ in trajectory
        }

        T = len(trajectory)

        for t in range(T):
            G   = 0.0
            rho = 1.0

            for k in range(t, T):
                state_k, action_k, reward_k = trajectory[k]

                # Cộng reward TRƯỚC — rho hiện tại áp cho reward bước k
                G += (self.gamma ** (k - t)) * rho * reward_k

                # Sau đó mới nhân rho cho bước tiếp theo
                best_k = greedy_snapshot.get(state_k, self.greedy_action(state_k))
                t_prob = 1.0 if action_k == best_k else 0.0
                b_prob = self._behaviour_prob(state_k, action_k)
                rho   *= t_prob / b_prob

                if rho < 1e-10:  # rho = 0 → các bước sau không đóng góp
                    break

            state, action_idx, _ = trajectory[t]
            self.C[state][action_idx] += 1
            self.Q[state][action_idx] += (
                (G - self.Q[state][action_idx])
                / self.C[state][action_idx]
            )
