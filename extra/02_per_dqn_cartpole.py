"""추가 실습 02 — 우선순위 재현 버퍼 (Prioritized Experience Replay, PER)

2일차 1교시에서 만든 재현 버퍼의 개선판입니다.

■ 무엇이 문제였나
기존 버퍼는 상자에서 **아무거나 똑같은 확률로** 꺼냈습니다.
그런데 경험마다 배울 게 다릅니다.

  "예상대로 흘러간 경험"  → 이미 아는 것. 다시 봐도 배울 게 없음
  "예상과 크게 달랐던 경험" → 놀라운 것. 여기서 배울 게 많음

교과서를 볼 때 이미 아는 페이지와 헷갈리는 페이지를 똑같이 반복하지 않는 것과 같습니다.
==헷갈리는 곳을 더 자주 보는 게 낫습니다.==

■ 어떻게 재나
"놀라움"을 숫자로 잽니다. 이미 배운 것이 있습니다. **TD 오차**입니다.
  오차 = (실제로 받은 것 + 다음 칸 값) − (내가 예상했던 값)
이 값이 크면 예상이 많이 빗나간 것이니, 그 경험을 더 자주 꺼냅니다.

■ 공짜는 아닙니다 — 편향이 생깁니다
자주 꺼내는 경험이 생기면 ==특정 경험에 치우쳐 배우게 됩니다.==
그래서 자주 뽑히는 경험은 학습할 때 **가중치를 낮춰** 보정합니다.
이것을 중요도 샘플링(importance sampling) 이라 부릅니다. 이름만 어렵습니다.

■ 실행
    python 02_per_dqn_cartpole.py
CPU 로 5~8분.

■ 실제로 돌려본 결과 (Intel CPU, 시드 고정)
400판에서 최고 359점까지 올라갔고 아직 오르는 중이었습니다.
CartPole 해결 기준(475)에는 못 미칩니다. EPISODES 를 600~800 으로 늘리면 넘습니다.
==여기서 중요한 건 "몇 점을 찍었나"가 아니라 "우선순위가 어떻게 작동하나"입니다.==
아래 [해보기] ①번으로 ALPHA=0 (기존 버퍼)과 비교해 보시는 것이 이 실습의 목적입니다.
"""
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

GAMMA = 0.99
BATCH = 64
LR = 1e-3
BUFFER_SIZE = 50_000
LEARN_START = 1_000
TARGET_SYNC = 500          # 스텝 단위 (2일차는 판 단위였습니다)
EPISODES = 400

ALPHA = 0.6    # 우선순위를 얼마나 세게 반영할지. 0이면 기존 버퍼와 같음
BETA0 = 0.4    # 편향 보정 강도. 학습이 진행되며 1.0까지 올립니다


class PrioritizedBuffer:
    """놀라웠던 경험을 더 자주 꺼내는 버퍼.

    구현은 단순하게 갔습니다. 실무에서는 합계 트리(sum tree)를 써서 빠르게 하지만,
    여기서는 numpy 로 확률을 직접 계산합니다. 원리를 보는 것이 목적입니다.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.prios = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, s, a, r, s2, done):
        # 새 경험은 "지금까지 중 가장 높은 우선순위"로 넣습니다.
        # 한 번은 반드시 뽑히게 해서, 아직 안 본 경험이 묻히지 않도록.
        max_prio = self.prios.max() if self.data else 1.0

        if len(self.data) < self.capacity:
            self.data.append((s, a, r, s2, done))
        else:
            self.data[self.pos] = (s, a, r, s2, done)

        self.prios[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        n = len(self.data)
        prios = self.prios[:n]

        # 우선순위를 확률로 바꿉니다.
        #   ALPHA=0 → 전부 같은 확률 (기존 버퍼와 동일)
        #   ALPHA=1 → 오차에 정비례
        probs = prios ** ALPHA
        probs /= probs.sum()

        idx = np.random.choice(n, batch_size, p=probs)

        # 편향 보정 가중치.
        # 자주 뽑히는 경험(probs 큰 것)은 가중치를 낮춰 균형을 맞춥니다.
        weights = (n * probs[idx]) ** (-beta)
        weights /= weights.max()          # 최대 1로 정규화 (크기 안정화)

        batch = [self.data[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        to = lambda x, t: torch.as_tensor(np.array(x), dtype=t, device=device)
        return (to(s, torch.float32), to(a, torch.int64), to(r, torch.float32),
                to(s2, torch.float32), to(d, torch.float32),
                idx, to(weights, torch.float32))

    def update_priorities(self, idx, errors):
        """학습해 보고 나온 새 오차로 우선순위를 갱신합니다."""
        # 1e-6 을 더하는 이유: 오차가 0이면 확률도 0이 되어
        # 그 경험이 영원히 안 뽑힙니다. 아주 작은 값을 남겨 둡니다.
        self.prios[idx] = np.abs(errors) + 1e-6

    def __len__(self):
        return len(self.data)


def mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, out_dim),
    )


def main():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = mlp(obs_dim, n_actions).to(device)
    q_target = mlp(obs_dim, n_actions).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

    buffer = PrioritizedBuffer(BUFFER_SIZE)
    eps, eps_min, eps_decay = 1.0, 0.05, 0.99
    returns_log = []
    step_count = 0

    for episode in range(EPISODES):
        state, _ = env.reset(seed=episode)
        total, done = 0.0, False

        while not done:
            # beta 를 0.4 → 1.0 으로 서서히 올립니다.
            # 초반에는 보정을 약하게(학습 속도 우선),
            # 후반에는 강하게(정확도 우선) 하는 것입니다.
            beta = min(1.0, BETA0 + (1.0 - BETA0) * step_count / 50_000)

            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = q_net(s_t).argmax().item()

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            buffer.push(state, action, reward, next_state, float(term))
            state = next_state
            total += reward
            step_count += 1

            if len(buffer) >= LEARN_START:
                s, a, r, s2, d, idx, w = buffer.sample(BATCH, beta)

                q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # Double DQN — 2일차 2교시에서 배운 것을 그대로 씁니다.
                    best_a = q_net(s2).argmax(1, keepdim=True)
                    q_next = q_target(s2).gather(1, best_a).squeeze(1)
                    target = r + GAMMA * q_next * (1 - d)

                td_error = target - q

                # 여기가 PER 의 두 번째 핵심입니다.
                # 손실에 가중치 w 를 곱합니다.
                # 자주 뽑히는 경험은 w 가 작아 학습에 덜 반영됩니다.
                loss = (w * td_error.pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

                # 방금 계산한 오차로 우선순위를 갱신합니다.
                # 잘 맞힌 경험은 우선순위가 내려가 덜 뽑히게 됩니다.
                buffer.update_priorities(idx, td_error.detach().cpu().numpy())

            if step_count % TARGET_SYNC == 0:
                q_target.load_state_dict(q_net.state_dict())

        eps = max(eps_min, eps * eps_decay)
        returns_log.append(total)

        if episode % 20 == 0:
            recent = np.mean(returns_log[-20:])
            print(f"판 {episode:3d}  최근 20판 평균 {recent:6.1f}  "
                  f"탐험률 {eps:.2f}  버퍼 {len(buffer)}")

        if len(returns_log) >= 20 and np.mean(returns_log[-20:]) >= 475:
            print(f"\nCartPole 해결! ({episode}판)")
            break

    env.close()
    return returns_log


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════
# 해보기
# ══════════════════════════════════════════════════════════
# ① ALPHA 를 0.6 → 0 으로 바꿔 보세요.
#    우선순위를 아예 안 쓰는 것이라 2일차의 기존 버퍼와 같아집니다.
#    두 결과를 비교하면 PER 의 효과를 직접 볼 수 있습니다.
#
# ② update_priorities 호출을 주석 처리해 보세요.
#    우선순위가 처음 값에 멈춰 있게 됩니다. 학습이 이상해지는 것을 볼 수 있습니다.
#
# ③ 가중치 w 를 곱하지 말고 그냥 td_error.pow(2).mean() 으로 바꿔 보세요.
#    편향 보정을 뺀 것입니다. 빠르지만 불안정해질 수 있습니다.
#
# ④ 1e-6 을 빼 보세요.
#    오차가 0인 경험이 영원히 안 뽑히게 되어, 나중에 그 상황을 잊어버립니다.
