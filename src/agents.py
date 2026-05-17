import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import numpy as np
from collections import defaultdict
from config import EPSILON, GAMMA, MAX_SPEED

ACTIONS = [(dvr, dvc) for dvr in [-1,0,1] for dvc in [-1,0,1]]

def valid_actions(vr: int, vc: int):
    result = []
    for i, (dvr, dvc) in enumerate(ACTIONS):
        new_vr = int(np.clip(vr+dvr, 0, MAX_SPEED))
        new_vc = int(np.clip(vc+dvc, 0, MAX_SPEED))
        if new_vr == 0 and new_vc==0:
            continue
        result.append(i)
    return result if result else [4] #action (0,0)

class MCAgent:
    def __init__(self, epsilon = EPSILON, gamma = GAMMA):
        self.epsilon = epsilon
        self.gamma = gamma

        self.Q = defaultdict(lambda: np.zeros(9))

        self.C = defaultdict(lambda: np.zeros(9))

    def greedy_action(self, state):
        _, _, vr, vc = state
        va = valid_actions(vr, vc)
        q_vals = self.Q[state]
        max_q = max(q_vals[i] for i in va)
        best = [i for i in va if q_vals[i] == max_q]

        return np.random.choice(best)   
    
    def epsilon_greedy(self, state):
        _, _, vr, vc = state
        va = valid_actions(vr, vc)
        if np.random.random() < self.epsilon:
            return np.random.choice(va)
        return self.greedy_action(state)
    
    def random_action(self, state):
        _,_, vr, vc = state
        va = valid_actions(vr, vc)
        return np.random.choice(va)
    
    def generate_episode(self, track, use_noise: bool = True):
        trajectory = []
        state = track.reset()

        for _ in range(1000):
            action_idx = self._select_action(state)
            action = ACTIONS[action_idx]
            next_state, reward, done,_ = track.step(state, action, use_noise)
            trajectory.append((state, action_idx, reward))
            if done:
                break
            state = next_state
        
        return trajectory
    
    def _select_action(self, state):
        return self.epsilon_greedy(state)
    
    def _update(self, trajectory):
        raise NotImplementedError
    
    def train_episode(self, track, use_noise = True):
        trajectory = self.generate_episode(track, use_noise)
        self._update(trajectory)
        return len(trajectory)
    

class OnPolicyAgent(MCAgent):
    def _update(self, trajectory):
        T = len(trajectory)
        G = 0.0
        returns = []
        for _, _, reward in reversed(trajectory):
            G = self.gamma * G + reward
            returns.append(G)
        returns.reverse()

        visited = set()
        for t, (state, acton_idx, _) in enumerate(trajectory):
            if (state, acton_idx) not in visited:
                visited.add((state, acton_idx))
                self.C[state][acton_idx] += 1
                self.Q[state][acton_idx] += ((returns[t] - self.Q[state][acton_idx])/self.C[state][acton_idx])

class OffPolicyAgent(MCAgent):
    def __init__(self, behaviour = 'random', **kwargs):
        super().__init__(**kwargs)
        assert behaviour in ('random', 'epsilon')
        self.behaviour = behaviour

    def _select_action(self, state):
        if self.behaviour == 'epsilon':
            return self.epsilon_greedy(state)
        return self.random_action(state)
    
    def _behaviour_prob(self, state: tuple, action_idx: int) -> float:
        _, _, vr, vc = state
        va = valid_actions(vr, vc)
        if self.behaviour == 'epsilon':
            best = self.greedy_action(state)
            if action_idx == best:
                return 1 - self.epsilon + self.epsilon / len(va)
            else:
                return self.epsilon / len(va)
        return 1.0 / len(va)  # uniform random
    
    def _target_prob(self, state: tuple, action_idx: int) -> float:
        best = self.greedy_action(state)
        return 1.0 if action_idx == best else 0.0
    
class WISAgent(OffPolicyAgent):
    def _update(self, trajectory):
        G = 0.0
        W = 1.0

        for state, action_idx, reward in reversed(trajectory):
            G = self.gamma*G + reward
            self.C[state][action_idx] += W
            self.Q[state][action_idx] += ((W / self.C[state][action_idx])
                                          * (G - self.Q[state][action_idx]))
            if action_idx != self.greedy_action(state):
                break
            b_prob = self._behaviour_prob(state, action_idx)
            W *= 1.0/b_prob

class OISAgent(OffPolicyAgent):
    def _update(self, trajectory):
        G =0.0
        W = 1.0
        for state, action_idx, reward in reversed(trajectory):
            G = self.gamma * G + reward

            # OIS: nhân thẳng G với W, dùng incremental mean đơn giản
            self.C[state][action_idx] += 1
            self.Q[state][action_idx] += (
                (W * G - self.Q[state][action_idx])
                / self.C[state][action_idx]
            )

            if action_idx != self.greedy_action(state):
                break

            b_prob = self._behaviour_prob(state, action_idx)
            W *= 1.0 / b_prob
    
class PerDecisionAgent(OffPolicyAgent):
    def _update(self, trajectory):
        T = len(trajectory)

        for t in range(T):
            G  = 0.0
            rho = 1.0  # IS ratio tích lũy từ bước t trở đi

            for k in range(t, T):
                state_k, action_k, reward_k = trajectory[k]

                # Tính rho cho bước k
                b_prob = self._behaviour_prob(state_k, action_k)
                t_prob = self._target_prob(state_k, action_k)
                rho   *= t_prob / b_prob

                # Cộng reward có chiết khấu và trọng số rho
                G += (self.gamma ** (k - t)) * rho * reward_k

                if rho < 1e-10:  # tránh nhân mãi với 0
                    break

            state, action_idx, _ = trajectory[t]
            self.C[state][action_idx] += 1
            self.Q[state][action_idx] += (
                (G - self.Q[state][action_idx])
                / self.C[state][action_idx]
            )

