# ==========================================================
# 2일차 4교시 — DQN, Double DQN 구현
# 2026-07-28 (화) 13:30 ~ 14:30 · Value-based & Policy-based Methods
# 원본 파일명: dqn_cartpole.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/2#s4
# ==========================================================
# [학습목표]
#  - CartPole-v1에서 DQN 전체 파이프라인을 완성한다
#  - 플래그 하나로 Double DQN으로 전환해 성능을 비교한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day2_전체.ipynb를 위에서부터 실행하세요.

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]     # 4
n_actions = env.action_space.n               # 2

q_net = QNetwork(obs_dim, n_actions)
q_target = QNetwork(obs_dim, n_actions)
q_target.load_state_dict(q_net.state_dict())
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer(50_000)

gamma, batch_size = 0.99, 64
eps, eps_min, eps_decay = 1.0, 0.05, 0.995
DOUBLE = True                                # ← Double DQN 스위치

def train_step():
    s, a, r, s_next, done = buffer.sample(batch_size)
    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if DOUBLE:   # 선택은 온라인넷, 평가는 타깃넷
            best_a = q_net(s_next).argmax(1, keepdim=True)
            q_next = q_target(s_next).gather(1, best_a).squeeze(1)
        else:        # 타깃넷이 선택+평가 (바닐라 DQN)
            q_next = q_target(s_next).max(1).values
        target = r + gamma * q_next * (1 - done)
    loss = nn.functional.smooth_l1_loss(q, target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

returns = []
for episode in range(400):
    s, _ = env.reset()
    total, done = 0, False
    while not done:
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = q_net(torch.as_tensor(s, dtype=torch.float32)).argmax().item()
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.push(s, a, r, s_next, float(term))
        s, total = s_next, total + r
        if len(buffer) >= 1000:
            train_step()
    eps = max(eps_min, eps * eps_decay)
    if episode % 20 == 0:                    # 타깃 네트워크 동기화
        q_target.load_state_dict(q_net.state_dict())
    returns.append(total)
    if episode % 20 == 0:
        print(f"ep {episode:3d}  return {np.mean(returns[-20:]):6.1f}  eps {eps:.2f}")
# 평균 리턴이 475를 넘으면 CartPole 해결!
